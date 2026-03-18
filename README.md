# Word2Vec

A word2vec implementation in pure Python/NumPy. This project was created as part of the application process for a JetBrains internship project, "Learning to Reason with Small Models".

The code was written with the help of the original word2vec paper (Mikolov et al., 2013, ICLR) and its later counterpart on improving efficiency with negative sampling and subsampling (Mikolov et al., 2013, NIPS).

## Training

After finishing the implementation, the next step was doing a training run. text8 was the recommended dataset for getting good results, so I tried that first. My laptop couldn't handle it though. After an hour of running it still wasn't finished and the CPU was sitting at around 98°C, so I had to stop.

To lessen the load, I switched to a smaller dataset called PTB (Penn Treebank), which is roughly 15x smaller (~700K tokens PTB vs text8 ~17M tokens). The results were disappointing. Even the classic analogy test (man → woman, king → ?) failed to produce anything meaningful.

At that point I wasn't really sure what to do. The task explicitly asked us to use NumPy and most likely didn't want us to go too far from that. So what I ended up doing is creating a separate file, `word2vec_njit.py`, which uses the Numba library to JIT-compile the training loop into optimized machine code that runs in parallel on the CPU. This brought training on text8 down to about 20 minutes on my laptop.

While working on the parallelized version, I noticed that the poor results on PTB might not have been caused by the smaller dataset size but by problems in the training logic itself. Unfortunately I ran out of time before I could go back and verify this, so it's something I'd still like to revisit.
