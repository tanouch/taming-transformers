# Goal 
The goal of the present page is to plug the architecture of VQGANs proposed by @lucidrains in the repo 
vector-quantize-pytorch (https://github.com/lucidrains/vector-quantize-pytorch) with the pipeline open-sourced by
@CompVis in the repo taming-transformers (https://github.com/CompVis/taming-transformers). 

Both repositery offer complementary features. Do not hesitate to look both pages for more details...

## How to run the new VQGAN architectures (prerequisite: ffhq dataset)
python main.py --base configs/ffhq_LucidrainsVQ.yaml -t True --gpus 0,
python main.py --base configs/ffhq_LucidrainsResidualVQ.yaml -t True --gpus 0,