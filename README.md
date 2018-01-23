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

```sh
usage: SoundTextureSynth.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH]
                            [-l OUTPUT_LENGTH] [-fs SAMPLE_RATE]
                            [-it ITER_TIME] [-lr LEARNING_RATE]
                            
arguments:
  -h
  -i
  -o
  -l
  -fs
  -it
  -lr
```


### Todos

 - Cross corelation faetures. 

License
----

BSD



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [test]: <http://dafx16.vutbr.cz/dafxpapers/19-DAFx-16_paper_18-PN.pdf>
