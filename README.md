# curated-links

Google Just released a Landmark recognition challenge on Kaggle, where you are given images of all major world landmarks and your CV algorithm learns to predict the landmark given a new image. The challenge is the number of landmarks (about 15k) and the limited training data for each landmark. This is the new ImageNet of the current generation. Wonderful to aim for
* [Link](https://www.kaggle.com/c/landmark-recognition-challenge)


DeepMind's new paper demonstrates how deep neural networks can be extended to generalize visually and symbolically.
* [Blog](https://deepmind.com/blog/learning-explanatory-rules-noisy-data/)
* [Paper](http://www.jair.org/media/5714/live-5714-10391-jair.pdf)

Google releases [AutoML Vision](https://www.blog.google/topics/google-cloud/cloud-automl-making-ai-accessible-every-business/), something that allows people to automatically solve vision problems with high accuracy with no ML expertise required!

Localize the object that is making sounds in a video. Trained from unlabelled video using only audio-visual correspondence as the objective function. A clever use of cross-modal self-supervision.
* Paper: https://arxiv.org/pdf/1712.06651.pdf
* Video: https://www.youtube.com/watch?v=TFyohksFd48

Mozilla has released open source speech recognition model of Baidu's deep speech & data (400k recordings, 500 hours of speech). Word error rate 6.5%, which is close to human.
* Data: https://voice.mozilla.org/data
* Tensorflow implementation: https://github.com/mozilla/DeepSpeech

An absoultely amazing paper from DeepMind on using a population based approach to tuning hyperparameters. The paper is really easy to understand and is useful for novices to experts. One of the best papers i've read in quite a bit of time. Hope you enjoy and I would love to see you use this in your projects.
* Paper: https://arxiv.org/abs/1711.09846
* Blog: https://deepmind.com/blog/population-based-training-neural-networks/

The Apple iPhone X face detection algorithm has been published by apple research. It's great to see Apple contributing back to the community. The algorithm itself is pretty standard, but is a good reference towards object detection in general. They use scaling, although there are better approaches today such as SSD, YOLO, but Apple might have gone with scaling to fit the performance into a mobile device. Hope you've a good read.
* Paper: https://machinelearning.apple.com/2017/11/16/face-detection.html

Google recently released SLING, an experimental system for parsing natural language text directly into a representation of a semantic frame graph. The output frame graph directly captures the semantic annotations of interest to the user. SLING uses a special-purpose recurrent neural network model to compute the output representation of input text. A really awesome step towards perfecting NLP!
* Link: https://research.googleblog.com/2017/11/sling-natural-language-frame-semantic.html

A summary of Capsule Net that was published last week. This is breaking all ground (and twitter too) on a brand new way to look at image processing. The article also has a code walkthrough of capsnet on tensorflow. If this is going above your head, no worries :). Just glance at it and bookmark, and you can get back to this at the end of the course.
* Link: https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc

Nervana Systems released Coach, a Deep RL library. Even if you don't use it, the docs are a great reference point.
* Link: http://coach.nervanasys.com/

A fantastic tutorial on unit testing machine learning code. The examples are in Tensorflow but the concepts can be applied for any ML framework out there. A must have when you are writing ML code.
* Blog: https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765

**A crazy application of CNNs + RNNs to choreograph dance steps**
"Dance Dance Revolution (DDR) is a popular rhythm-based video game. Players perform steps on a dance platform in synchronization with music as directed by on-screen step charts. While many step charts are available in standardized packs, players may grow tired of existing charts, or wish to dance to a song for which no chart exists. We introduce the task of learning to choreograph. Given a raw audio track, the goal is to produce a new step chart. This task decomposes naturally into two subtasks: deciding when to place steps and deciding which steps to select. For the step placement task, we combine recurrent and convolutional neural networks to ingest spectrograms of low-level audio features to predict steps, conditioned on chart difficulty. For step selection, we present a conditional LSTM generative model that substantially outperforms n-gram and fixed-window approaches."
* Paper: https://arxiv.org/abs/1703.06891

A fun project to solve a rubicks cube using a convolutional neural net.
* Link: https://github.com/jerpint/rubiks_cube_convnet

Tensorflow is one of the most popular deep learning frameworks and it incorporates TensorBoard, a great visualizer to debug your models. But using Tensorboard isn't always obvious and this nice article does a great job explaining the "how to".
* Blog: https://medium.com/towards-data-science/visualizing-your-model-using-tensorboard-796ebb73e98d

EuroPilot is a AI connector to the popular Euro Truck simulator. You can try a full fledged self driving car within the game. They've already implemented an end to end DL training based on nVidia's model. You can modify this and play around with a lot of things.
* Link: https://github.com/marshq/europilot

MIT's deep learning for Self Driving Cars course is their take on term 1. Covers DL is detail, and includes reinforcement learning too! Definitely worth a watch
* Link: https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf

With autonomous driving slowly taking over, most car manufactures are still in the process and the process could typically take a decade before fully autonomous cars comes in. Until then, it's only going to be partially autonomous, where the car might have to hand over to the driver periodically when it runs into a difficult situation (snow, etc...). At such points, it's very important for the car to ensure (constantly) that the driver is alert. A great application here is drowsiness / fatigue detection using deep learning and computer vision. 
* Link: https://www.youtube.com/watch?v=uGvvCgqYmpY

A fantastic survey on various deep learning techniques used in building AIs for video gaming, starting from minecraft to Blizzard Entertainment's Starcraft II. Hope you enjoy :).
* Paper: https://arxiv.org/pdf/1708.07902.pdf

"With so much at stake, the need for the field of neuroscience and AI to come together is now more urgent than ever before."
* Article: https://deepmind.com/blog/ai-and-neuroscience-virtuous-circle/

DeepMoji: Predicting emojis for classifying text sentiment/emotion/sarcasm. You can see how it get's the sentiment/tone of the sentence even though the words are very similar. Especially love how it distinguishes between "This is shit" and "This is the shit" :)
* Paper: https://arxiv.org/abs/1708.00524

Google recently released a tool called Facets! It's a Data analysis tool used to understand, visualize and debug ML datasets. Data is THE most important thing in machine learning and having a tool to visualize and debug the data is absolutely fantastic.
* Demo website: https://pair-code.github.io/facets/ 
* Blog: https://research.googleblog.com/2017/07/facets-open-source-visualization-tool.html 
* Code: https://github.com/pair-code/facets

nSynth, a magenta (google)'s project on neural music synthesis. They also have a huge set of data, models, documentation. Great to read up in general on generative networks. https://magenta.tensorflow.org/nsynth

So looks like finally Apple is starting to publish their ML research. About high time they contribute back to the industry and this is a good start. Another link to bookmark and periodically watch out for articles. https://machinelearning.apple.com/

An excellent tutorial on over / under fitting a dataset and how it impacts a real life situation (Fukushima disaster) from Berkeley. Balancing fitting is the one key skill that any ML engineer needs. https://ml.berkeley.edu/blog/2017/07/13/tutorial-4/

A very nice video on how a neural net learns fluid dynamics :) https://www.youtube.com/watch?v=iOWamCtnwTc&app=desktop. Hope you enjoy.

A fantastic write-up on a solution to lung cancer detection from 3D CT scans. Inference from 3D systems are significantly more complicated compared to 2D (say images), so this is a very interesting write-up. Also it covers a lot of fundamentals on machine learning, such as handling data (preprocessing), model selection, along with the code itself. Hope you've fun http://juliandewit.github.io/kaggle-ndsb2017/

"In this paper we cluster 330 classical music pieces collected from MusicNet database based on their musical note sequence. We use shingling and chord trajectory matrices to create signature for each music piece and performed spectral clustering to find the clusters. Based on different resolution, the output clusters distinctively indicate composition from different classical music era and different composing style of the musicians." https://arxiv.org/abs/1706.08928

"We carve our world into relations between things. This ability is called relational reasoning and is central to human intelligence. A key challenge in developing artificial intelligence systems with the flexibility and efficiency of human cognition is giving them a similar ability - to reason about entities and their relations from unstructured data." Fantastic research article / paper from DeepMind (https://deepmind.com/blog/neural-approach-relational-reasoning/).

Andrej Karpathy's (One of the world's leading deep learning expert) take on the recent Alpha Go's success against world Number 1. A wonderful read for all AI / ML enthusiasts :). Covers Monte Carlo Trees, Reinforcement Learning, Supervised Learning, etc... Hope you've fun reading this: https://medium.com/@karpathy/alphago-in-context-c47718cb95a5

Hey. A great new development. A ML model to convert from a GUI (an image) to code :). AI is entering front end stack space and like always, is going to disrupt humans and existing companies. https://arxiv.org/abs/1705.07962. Have fun reading it.

Great couple of articles that introduces various start-ups in the autonomous industry and the trend within the industry to partner up and collaborate on product development. Absolutely important insight for those who want to create their own startup / join one and to generally understand the industry. https://blog.cometlabs.io/263-self-driving-car-startups-to-watch-8a9976dc62b0 http://www.sfchronicle.com/business/article/Partner-up-Self-driving-car-firms-form-tangled-11160522.php

Using Machine Learning to create a neural net is a great way to tune the hyper-parameters automatically! It could also be a generic framework that can just spit out neural nets for each specific problem. Google seems to be making some headway here. Fantastic paper and article. Let me know your thoughts (https://research.googleblog.com/2017/05/using-machine-learning-to-explore.html)

An exceptionally amazing article introducing unsupervised learning and the future of Deep Learning. Many including Yann LeCunn, Andrew Ng, and our own Sebastian Thrun think that unsupervised learning is the future, simply because of how elegantly it avoids the logistics of data collection and annotation. Hope you like the read (https://medium.com/intuitionmachine/navigating-the-unsupervised-learning-landscape-951bd5842df9)

If you've always wondered how Google keeps maps updated, the answer is a lot of manual inputs via "Google local guides" and in many places Deep learning clubbed with street view. An article / paper discussing the challenges along with their solution. Wonderful to read solutions to real life problems: https://research.googleblog.com/2017/05/updating-google-maps-with-deep-learning.html. Have fun reading :)

This is a fantastic paper from MIT / GOOGLE. It talks about how neural nets generalise. If you are not yet familiar with neural nets and CNNs, I'd suggest that you bookmark this paper and read it after you gain an understanding of NNs. Understanding how neural nets generalize is a key concept to understand how to push NNs into the next generation, and in that I felt this paper is excellent! https://openreview.net/pdf?id=Sy8gdB9xx

So i'm going to share a philosophical article now :). Trying to define consciousness. And this is a very interesting topic, because essentially with deep learning / AI / Robotics, we are essentially trying to mimic humans, and consciousness is a very important part of that. So read, enjoy and let me know what you think about this... http://www.bbc.com/news/science-environment-39482345 Oh and if you haven't watched westworld yet, please do :). If you've, then that's one more interesting thing we can talk about.

Currently almost all ML training comes from supervised data, data probably collected from human interactions. Which means that the ML almost certainly learns all the biases we operate with and this is a major issue in supervised learning. Great article to read more about this http://www.geekwire.com/2017/scientists-zero-artificial-intelligence-absorbs-human-biases/

After we put in all this intelligence in the car, how easy is it going to be for hackers to hack the car and cause mayhem. We need really good security systems on board. https://www.wired.com/2017/04/ubers-former-top-hacker-securing-autonomous-cars-really-hard-problem/

Going about solving social problems using machine learning from Harvard. Not sure if you've already seen this, but it's a fantastic read on how what we are learning might apply beyond conventional engineering https://hbr.org/2016/12/a-guide-to-solving-social-problems-with-machine-learning

A fantastic overview on their SDC from Ford! https://medium.com/@ford/what-it-takes-to-be-a-self-driving-leader-71928b94870e

An AI that uses neural networks to write programs and solve programming challenges. This is a fantastic step forward because it's the start of programs being able to write out simpler programs (https://blog.acolyer.org/2017/03/29/deepcoder-learning-to-write-programs/)

Hey! Udacity just opened it's hiring portal. :D If you are interested, I would suggest that you bookmark it and periodically check and apply to positions that you are interested in. https://career-resource-center.udacity.com/hiring-partners-jobs

Astronomers using AI generated images (using deep learning): http://www.nature.com/news/astronomers-explore-uses-for-ai-generated-images-1.21398

Stanford unveils their entire NLP (Natural Language Processing) tutorials online: https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6

Going about solving social problems using machine learning from Harvard: https://hbr.org/2016/12/a-guide-to-solving-social-problems-with-machine-learning

A great article on Chess AI. They are calling it the hello world for AI: https://medium.freecodecamp.com/simple-chess-ai-step-by-step-1d55a9266977

Currently almost all ML training comes from supervised data, data probably collected from human interactions. Which means that the ML almost certainly learns all the biases we operate with and this is a major issue in supervised learning. Great article to read more about this http://www.geekwire.com/2017/scientists-zero-artificial-intelligence-absorbs-human-biases/

Trying to define consciousness. And this is a very interesting topic, because essentially with deep learning / AI / Robotics, we are essentially trying to mimic humans, and consciousness is a very important part of that: http://www.bbc.com/news/science-environment-39482345

This is a fantastic paper from MIT / GOOGLE. It talks about how neural nets generalize. Understanding how neural nets generalize is a key concept to understand how to push NNs into the next generation: https://openreview.net/pdf?id=Sy8gdB9xx

If you've always wondered how Google keeps maps updated, the answer is a lot of manual inputs via "Google local guides" and in many places Deep learning clubbed with street view: https://research.googleblog.com/2017/05/updating-google-maps-with-deep-learning.html

An exceptionally amazing article introducing unsupervised learning and the future of Deep Learning: https://medium.com/intuitionmachine/navigating-the-unsupervised-learning-landscape-951bd5842df9

Using Machine Learning to create a neural net is a great way to tune the hyper-parameters automatically! It could also be a generic framework that can just spit out neural nets for each specific problem. Google seems to be making some headway here. Fantastic paper and article: https://research.googleblog.com/2017/05/using-machine-learning-to-explore.html

A great introduction on using Tensorflow: https://www.youtube.com/watch?v=5DknTFbcGVM&t

Cool piece of code using tensorflow / Keras and OpenCV to do emotion / gender classification from the face of a person. Very interesting work. https://github.com/oarriaga/face_classification

Great couple of articles that introduces various start-ups in the autonomous industry and the trend within the industry to partner up and collaborate on product development: 
o	https://blog.cometlabs.io/263-self-driving-car-startups-to-watch-8a9976dc62b0
o	http://www.sfchronicle.com/business/article/Partner-up-Self-driving-car-firms-form-tangled-11160522.php

After we put in all this intelligence in the car, how easy is it going to be for hackers to hack the car and cause mayhem. We need really good security systems on board. https://www.wired.com/2017/04/ubers-former-top-hacker-securing-autonomous-cars-really-hard-problem/

A fantastic overview on their SDC from Ford! https://medium.com/@ford/what-it-takes-to-be-a-self-driving-leader-71928b94870e

An AI that uses neural networks to write programs and solve programming challenges. This is a fantastic step forward because it's the start of programs being able to write out simpler programs https://blog.acolyer.org/2017/03/29/deepcoder-learning-to-write-programs/
