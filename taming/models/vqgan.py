import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import VectorQuantizeLucidrains, ResidualVQLucidrains

from torchmetrics import Metric
import numpy as np
import matplotlib.pyplot as plt

class MyCount(Metric):
    def __init__(self, vocab_size, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("count", default=torch.zeros((vocab_size)).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros((1)).float(), dist_reduce_fx="sum")

    def update(self, z_indices: torch.Tensor):
        with torch.no_grad():
            z_indices = z_indices.view(-1).detach()
            self.count.index_add_(0,z_indices,torch.ones_like(z_indices).type_as(self.count))
            self.total += z_indices.shape[0]

    def compute(self):
        return self.count/self.total

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 normalize_embedding=False,
                 lr_disc_factor = 1.,
                 splitting = False,
                 split_frequency = 2000,
                 total_count_reset = True,
                 split_most_dist=False,
                 discard_tokens_freq = 1e-7,
                 continuous=False,
                 legacy = None,
                 splitting_scheduling=None,
                 power_val=2,
                 power_nb=10
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.normalize_embedding = normalize_embedding
        self.lr_disc_factor = lr_disc_factor
        self.total_count = MyCount(n_embed)
        self.batch_count = MyCount(n_embed)
        self.split_frequency = split_frequency
        self.discard_tokens_freq = discard_tokens_freq
        self.total_count_reset = total_count_reset
        self.splitting = splitting
        self.n_embed = n_embed
        self.continuous = continuous
        if self.continuous:
            beta = 0.
            if legacy is None:
                legacy = False
            else:
                legacy = legacy
        else:
            beta = 0.25
            if legacy is None:
                legacy = True
            else:
                legacy = legacy
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=beta,
                                        remap=remap, sane_index_shape=sane_index_shape,
                                        splitting=splitting,split_most_dist=split_most_dist,
                                        discard_tokens_freq=discard_tokens_freq, legacy=legacy)

        if splitting_scheduling is None:
            self.split_values = [i for i in range(1000000) if i%self.split_frequency==0 and i!=0]
        else:
            if splitting_scheduling == 'power':
                self.split_values = [int(i) for i in np.power(power_val,np.arange(power_nb)+2)]
                self.split_values += [i for i in range(1000000) if i%self.split_frequency==0 and i!=0]
            else:
                print('!!!! Scheduling not implemented !!!!')
                raise NotImplementedError

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            #self.load_from_checkpoint(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, quantize_continous=False):
        if self.normalize_embedding:
            with torch.no_grad():
                #norms = torch.norm(self.quantize.embedding.weight, p=2, dim=1).data
                #self.quantize.embedding.weight.data = \
                #    self.quantize.embedding.weight.data.div(norms.unsqueeze(1).expand_as(self.quantize.embedding.weight))
                self.quantize.embedding.weight.data = F.normalize(self.quantize.embedding.weight,dim=1)


        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.normalize_embedding:
            #norms = torch.norm(h, p=2, dim=1,keepdim=True)
            #h = h / (norms + 1e-8)
            h = F.normalize(h,dim=1)
        quant, emb_loss, info = self.quantize(h)
        if self.continuous and not quantize_continous:
            quant = h
            emb_loss = emb_loss
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, quantize_continous=False):
        quant, diff, (_,_,min_encoding_indices) = self.encode(input, quantize_continous)
        dec = self.decode(quant)
        return dec, diff, min_encoding_indices

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            if self.global_step in self.split_values:
                this_total_count = self.total_count.compute()
                if self.splitting:
                    self.quantize.split(this_total_count)
                plt.clf()
                plt.plot(np.arange(self.n_embed),-np.sort(-self.total_count.count.detach().cpu().numpy()))
                plt.savefig('histograms/hist_'+str(batch_idx))
                if self.total_count_reset:
                    self.total_count.reset()

        x = self.get_input(batch, self.image_key)
        xrec, qloss, quant = self(x)

        if optimizer_idx == 0:
            # autoencode
            self.batch_count.update(quant)
            batch_count = torch.count_nonzero((self.batch_count.count))
            self.batch_count.reset()
            self.total_count.update(quant)
            total_count_compute = self.total_count.compute()
            total_count = torch.count_nonzero((total_count_compute))
            freq_count = torch.count_nonzero(total_count_compute > self.discard_tokens_freq)
            self.log("train/frequency_counter", freq_count.float(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/batch_counter", batch_count.float(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/total_counter", total_count.float(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, quant = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        #rec_loss = log_dict_ae["val/rec_loss"]
        #self.log("val/rec_loss", rec_loss,
        #           prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, on_step=True, on_epoch=True, prog_bar=True, \
                                        logger=True, batch_size=x.shape[0])
        #self.log("val/aeloss", aeloss,
        #           prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, on_step=True, on_epoch=True, prog_bar=True, \
                                    logger=True, batch_size=x.shape[0])

        if self.continuous:
            xrec_q, _, _ = self(x, quantize_continous=True)
            aeloss_q, log_dict_ae_q = self.loss(qloss, x, xrec_q, 0, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")
            rec_loss_q = log_dict_ae_q["val/rec_loss"]
            self.log("val/rec_loss_q", rec_loss_q,
                       prog_bar=True, logger=True, on_step=True, on_epoch=True,
                        sync_dist=True, batch_size=x.shape[0])

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.lr_disc_factor * lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _ = self(x)
        if self.continuous:
            xrec_q, _, _ = self(x, quantize_continous=True)
            log["reconstructions_q"] = xrec_q
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class VQLucidrainsModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 dim = 256,
                 codebook_dim = None,
                 codebook_size = None,
                 kmeans_init = True,
                 kmeans_iters = 1,
                 use_cosine_sim = True, 
                 threshold_ema_dead_code = 1, 
                 accept_image_fmap=False
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = VectorQuantizeLucidrains(
            dim = dim,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            use_cosine_sim = use_cosine_sim, 
            threshold_ema_dead_code = threshold_ema_dead_code,
            accept_image_fmap = accept_image_fmap
        )
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

            
class ResVQLucidrainsModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 dim = 256,
                 codebook_dim = None,
                 codebook_size = None,
                 kmeans_init = True,
                 kmeans_iters = 1,
                 use_cosine_sim = True, 
                 threshold_ema_dead_code = 1, 
                 accept_image_fmap=False, 
                 num_quantizers = None
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = ResidualVQLucidrains(
            dim = dim,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            use_cosine_sim = use_cosine_sim, 
            threshold_ema_dead_code = threshold_ema_dead_code,
            accept_image_fmap = accept_image_fmap, 
            num_quantizers = num_quantizers
        )
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)