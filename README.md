# Word2Vec

A word2vec implementation in pure Python/NumPy. This project was created as part of the application process for a JetBrains internship project, "Learning to Reason with Small Models".

The code was written with the help of the original word2vec paper (Mikolov et al., 2013, ICLR) and its later counterpart on improving efficiency with negative sampling and subsampling (Mikolov et al., 2013, NIPS).

## Training

After finishing the implementation, the next step was doing a training run. text8 was the recommended dataset for getting good results, so I tried that first. My laptop couldn't handle it though. After an hour of running it still wasn't finished and the CPU was sitting at around 98°C, so I had to stop.

To lessen the load, I switched to a smaller dataset called PTB (Penn Treebank), which is roughly 15x smaller (~700K tokens PTB vs text8 ~17M tokens). The results were disappointing. Even the classic analogy test (man → woman, king → ?) failed to produce anything meaningful.

At that point I wasn't really sure what to do. The task explicitly asked us to use NumPy and most likely didn't want us to go too far from that. So what I ended up doing is creating a separate file, `word2vec_njit.py`, which uses the Numba library to JIT-compile the training loop into optimized machine code that runs in parallel on the CPU. This brought training on text8 down to about 20 minutes on my laptop.

While working on the parallelized version, I noticed that the poor results on PTB might not have been caused by the smaller dataset size but by problems in the training logic itself. Unfortunately I ran out of time before I could go back and verify this, so it's something I'd still like to revisit.

## Results

dataset: text8
embedding dimensions: 100
training epochs: 5


```
nearest neighbors
--------------------------------------------------
  king         -> lulach (0.798), pretender (0.789), eochaid (0.785), plantagenet (0.784), canute (0.780)
  queen        -> elizabeth (0.789), highness (0.764), regnant (0.755), consort (0.753), thrones (0.741)
  man          -> shameless (0.676), luckiest (0.674), wise (0.672), cowardly (0.670), gracious (0.669)
  woman        -> prostitute (0.701), she (0.695), husbands (0.691), sex (0.685), intercourse (0.675)
  france       -> belgium (0.740), nantes (0.724), netherlands (0.712), napol (0.704), trondheim (0.699)
  paris        -> cimeti (0.729), rodin (0.725), villa (0.722), universelle (0.720), chapelle (0.713)
  computer     -> computers (0.812), microcomputers (0.772), bootstrap (0.772), computing (0.769), pdas (0.764)
  good         -> everyone (0.689), honestly (0.684), cares (0.683), appreciate (0.681), surely (0.679)
  day          -> celebrates (0.778), thanksgiving (0.775), candlemas (0.774), holiday (0.757), monday (0.750)
  one          -> seven (0.896), six (0.885), three (0.879), eight (0.868), five (0.858)
  war          -> bloodiest (0.774), armistice (0.742), yalta (0.740), irregulars (0.737), invades (0.735)

analogies  (a:b :: c:?)
--------------------------------------------------
  man:woman :: king:? -> marrying (0.668), empress (0.659), concubine (0.654)
  man:woman :: uncle:? -> wife (0.676), aunt (0.645), remarried (0.638)
  france:paris :: germany:? -> dresden (0.742), munich (0.741), berlin (0.735)
  big:bigger :: small:? -> larger (0.668), smaller (0.657), pods (0.614)
  go:going :: play:? -> plays (0.632), boitano (0.628), calvinball (0.625)
```