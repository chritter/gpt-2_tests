# Experiment with Sentiment Data

Adaption of this example from  https://gmihaila.medium.com/gpt2-for-text-classification-using-hugging-face-transformers-574555451832


* Models 
    * gpt2-xl
        * 1557614400 parameters, ~6GB large
        * CUDA out of memory. Tried to allocate 48.00 MiB (GPU 0; 15.78 GiB total capacity;
        * 48 attention heads
        * during batch loading or model loading?
        * on cpu modeling takes 20GB+. epoch would taek 64hrs 
    * gpt2 
        * number of parameters: 124 441 344
        * [Pure model GPU memory: 7469 / 16160 MB
        * 12 attention heads
    * gpt2-large
        *  Tried to allocate 38.00 MiB (GPU 0; 15.78 GiB total capacity; 14.10 GiB already allocated
        * 774 032 640 params
    * gpt2-medium:
        * 354 825 216 params
        * runs on V100! 
        * 24 attention heads
        * try to increase batch size from 1, 8 does not work.
        * ~1hr/batch
