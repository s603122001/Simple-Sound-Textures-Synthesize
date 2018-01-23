# Simple Sound Textures Synthesize


Using statistic methods to synthesize sound of textures from white noise.

This is a simple implementation for the algorithm described in 

*Syhthesis of sound textures with tonal components using summary statistic and all-pole residual modeling*<br/>
*Hyung-Suk, Kim and Julius Smith*<br/>
*DAFx-16*<br/>


### Dependencies

This repository requires following packages:

- python 2.7
- numpy
- essentia
- librosa
- gammatone

### Usage

```
usage: SoundTextureSynth.py [-h] [-i INPUT_PATH] [-o OUTPUT_NAME]
                            [-l OUTPUT_LENGTH] [-fs SAMPLE_RATE]
                            [-it ITER_TIME] [-lr LEARNING_RATE]
                            
required arguments:
  -i  INPUT_PATH     path to input file (source audio)
  
optional arguments:
  -h
  -o  OUTPUT_NAME    name of output file (default = 'out.wav')
  -l  OUTPUT_LENGTH  length of output file(in seconds) (default = 5)
  -fs SAMPLE_RATE    sample rate (default = 44100)
  -it ITER_TIME      Maximum iteration time for gradient decent(in seconds) (default = 60)
  -lr LEARNING_RATE  learning rate for gradient decent (default = 0.3)
```


### Todos

 - Cross corelation faetures. 

License
----

BSD



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [test]: <http://dafx16.vutbr.cz/dafxpapers/19-DAFx-16_paper_18-PN.pdf>
