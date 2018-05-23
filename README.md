# curated-links

## Streams of Deep Learning
### Unsupervised Learning
* An exceptionally amazing [article](https://medium.com/intuitionmachine/navigating-the-unsupervised-learning-landscape-951bd5842df9) introducing unsupervised learning and the future of Deep Learning. Many including Yann LeCunn, Andrew Ng, and Sebastian Thrun think that unsupervised learning is the future, simply because of how elegantly it avoids the logistics of data collection and annotation.

### Reinforcement Learning
* Open AI has announced the next set of [research problems](https://blog.openai.com/requests-for-research-2/) that they want to solve. This is a fantastic way to get into deep learning and reinforcement learning and really push your boundaries. And you'll be working along side a really amazing community, so the learning opportunities are fantastic
* Nervana Systems released [Coach](http://coach.nervanasys.com/), a Deep RL library.
* In this [blog post](https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html) series Johannes takes a closer look at inverse reinforcement learning (IRL) which is the field of learning an agent's objectives, values, or rewards by observing its behavior.

### Natural Language Processing (NLP)
* And a [remote NLP study group](http://forums.fast.ai/t/remote-nlp-focused-study-group/10768) has started. A great way to meet like minded folks and learn more about NLP if you are interested. Link: 
* Mozilla has released open source speech recognition [model](https://github.com/mozilla/DeepSpeech) of Baidu's deep speech & [data](https://voice.mozilla.org/data) (400k recordings, 500 hours of speech). Word error rate 6.5%, which is close to human.
* Google released [SLING](https://research.googleblog.com/2017/11/sling-natural-language-frame-semantic.html), an experimental system for parsing natural language text directly into a representation of a semantic frame graph. The output frame graph directly captures the semantic annotations of interest to the user. SLING uses a special-purpose recurrent neural network model to compute the output representation of input text. A really awesome step towards perfecting NLP!
* [DeepMoji](https://arxiv.org/abs/1708.00524): Predicting emojis for classifying text sentiment/emotion/sarcasm. You can see how it get's the sentiment/tone of the sentence even though the words are very similar. Especially love how it distinguishes between "This is shit" and "This is the shit" :)
* Stanford unveils their entire NLP (Natural Language Processing) [tutorials](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) online.

### Computer Vision
* Google released a [Landmark recognition](https://www.kaggle.com/c/landmark-recognition-challenge) challenge on Kaggle, where you are given images of all major world landmarks and your CV algorithm learns to predict the landmark given a new image. The challenge is the number of landmarks (about 15k) and the limited training data for each landmark. This is the new ImageNet of the current generation.
* Google releases [AutoML Vision](https://www.blog.google/topics/google-cloud/cloud-automl-making-ai-accessible-every-business/), something that allows people to automatically solve vision problems with high accuracy with no ML expertise required!
* Localize the object that is making sounds in a video. Trained from unlabelled video using only audio-visual correspondence as the objective function. A clever use of cross-modal self-supervision: Paper: https://arxiv.org/pdf/1712.06651.pdf; Video: https://www.youtube.com/watch?v=TFyohksFd48
* The Apple iPhone X [face detection](https://machinelearning.apple.com/2017/11/16/face-detection.html) algorithm has been published by apple research. It's great to see Apple contributing back to the community. The algorithm itself is pretty standard, but is a good reference towards object detection in general. They use scaling, although there are better approaches today such as SSD, YOLO, but Apple might have gone with scaling to fit the performance into a mobile device.
* A summary of [Capsule Net](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc). The article also has a code walkthrough of capsnet on tensorflow.
* Neural net [learns](https://arxiv.org/abs/1703.06891) "Dance Dance Revolution (DDR)"
* A fun [project](https://github.com/jerpint/rubiks_cube_convnet) to solve a rubicks cube using a convolutional neural net.
* Cool piece of [code](https://github.com/oarriaga/face_classification) using tensorflow / Keras and OpenCV to do emotion / gender classification from the face of a person.
* And a recent [blog](https://research.googleblog.com/2018/03/using-evolutionary-automl-to-discover.html) from Google brain talks about allowing CNNs to evolve on their own and various different approaches we can take, so as to automatically generate models without having to hand craft or minimally hand craft things.

## Generic
* DeepMind's new paper demonstrates how deep neural networks can be extended to generalize visually and symbolically. [Blog](https://deepmind.com/blog/learning-explanatory-rules-noisy-data/); [Paper](http://www.jair.org/media/5714/live-5714-10391-jair.pdf)
* An absoultely amazing [paper](https://arxiv.org/abs/1711.09846) from DeepMind on using a population based approach to tuning hyperparameters. The paper is really easy to understand and is useful for novices to experts. The [blog](https://deepmind.com/blog/population-based-training-neural-networks/) is also available for an easy read.
* A fantastic [tutorial](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) on unit testing machine learning code. The examples are in Tensorflow but the concepts can be applied for any ML framework out there. A must have when you are writing ML code.
* A great article on the [interactions](https://deepmind.com/blog/ai-and-neuroscience-virtuous-circle/) between neuroscience and AI from DeepMind
* Google recently released a tool called Facets! It's a Data analysis tool used to understand, visualize and debug ML datasets. Data is THE most important thing in machine learning and having a tool to visualize and debug the data is absolutely fantastic: 
[Demo website](https://pair-code.github.io/facets/); [Blog](https://research.googleblog.com/2017/07/facets-open-source-visualization-tool.html); [Code](https://github.com/pair-code/facets)
* An excellent [tutorial](https://ml.berkeley.edu/blog/2017/07/13/tutorial-4/) on over / under fitting a dataset and how it impacts a real life situation (Fukushima disaster) from Berkeley. Balancing fitting is the one key skill that any ML engineer needs.
* "We carve our world into relations between things. This ability is called relational reasoning and is central to human intelligence. A key challenge in developing artificial intelligence systems with the flexibility and efficiency of human cognition is giving them a similar ability - to reason about entities and their relations from unstructured data." Fantastic research [article](https://deepmind.com/blog/neural-approach-relational-reasoning/) / paper from DeepMind.
* Using Machine Learning to create a neural net is a great way to tune the hyper-parameters automatically! It could also be a generic framework that can just spit out neural nets for each specific problem. Google seems to be making some headway here. Fantastic [article](https://research.googleblog.com/2017/05/using-machine-learning-to-explore.html)
* This is a fantastic [paper](https://openreview.net/pdf?id=Sy8gdB9xx) from MIT / GOOGLE. It talks about how neural nets generalise. If you are not yet familiar with neural nets and CNNs, I'd suggest that you bookmark this paper and read it after you gain an understanding of NNs. Understanding how neural nets generalize is a key concept to understand how to push NNs into the next generation, and in that I felt this paper is excellent! 
* Trying to define [consciousness](http://www.bbc.com/news/science-environment-39482345). And this is a very interesting topic, because essentially with deep learning / AI / Robotics, we are essentially trying to mimic humans, and consciousness is a very important part of that.
* Tensorflow is one of the most popular deep learning frameworks and it incorporates TensorBoard, a great visualizer to debug your models. But using Tensorboard isn't always obvious and this nice [article](https://medium.com/towards-data-science/visualizing-your-model-using-tensorboard-796ebb73e98d) does a great job explaining the "how to".
* Google introduced ["Learn with Google AI"](https://ai.google/education#?modal_active=none) — a set of educational resources developed by Machine Learning (ML) experts at the company, for people to learn about concepts, develop skills and apply Artificial Intelligence (AI) to real-world problems.
* A really awesome product from Deep Cognition, the [deep learning studio](http://deepcognition.ai/desktop/) includes features like a full-featured GUI model editor, graphical training dashboard. And the best part is that it's free.

## Applications
### Self Driving Cars
* [EuroPilot](https://github.com/marshq/europilot) is a AI connector to the popular Euro Truck simulator. You can try a full fledged self driving car within the game. They've already implemented an end to end DL training based on nVidia's model. You can modify this and play around with a lot of things.
* MIT's deep learning for [Self Driving Cars course](https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf) is their take on term 1. Covers DL is detail, and includes reinforcement learning too! Definitely worth a watch
* With autonomous driving slowly taking over, most car manufactures are still in the process and the process could typically take a decade before fully autonomous cars comes in. Until then, it's only going to be partially autonomous, where the car might have to hand over to the driver periodically when it runs into a difficult situation (snow, etc...). At such points, it's very important for the car to ensure (constantly) that the driver is alert. A great [application](https://www.youtube.com/watch?v=uGvvCgqYmpY) here is drowsiness / fatigue detection using deep learning and computer vision.
* After we put in all this intelligence in the car, how easy is it going to be for hackers to hack the car and cause mayhem. We need really good [security systems](https://www.wired.com/2017/04/ubers-former-top-hacker-securing-autonomous-cars-really-hard-problem/) on board.
* A fantastic [overview](https://medium.com/@ford/what-it-takes-to-be-a-self-driving-leader-71928b94870e) on their SDC from Ford!
* An amazing [talk](https://www.youtube.com/watch?v=LSX3qdy0dFg) by Sacha Arnoud (Directory of Engineering at Waymo) at MIT titled "The rise of machine learning in self-driving cars."

### Gaming
* A fantastic [survey](https://arxiv.org/pdf/1708.07902.pdf) on various deep learning techniques used in building AIs for video gaming, starting from minecraft to Blizzard Entertainment's Starcraft II.
* Andrej Karpathy's (One of the world's leading deep learning expert) take on the recent [Alpha Go](https://medium.com/@karpathy/alphago-in-context-c47718cb95a5)'s success against world Number 1. A wonderful read for all AI / ML enthusiasts :). Covers Monte Carlo Trees, Reinforcement Learning, Supervised Learning, etc...
* A great [article](https://medium.freecodecamp.com/simple-chess-ai-step-by-step-1d55a9266977) on Chess AI. They are calling it the hello world for AI.
* Using GAN to [generate](https://www.fastcompany.com/40568981/the-machines-have-taught-themselves-to-make-mario-levels) Mario levels! Hop, hop only to find out the princess isn't in the castle!

### Music
* [nSynth](https://magenta.tensorflow.org/nsynth), a magenta (google)'s project on neural music synthesis. They also have a huge set of data, models, documentation. Great to read up in general on generative networks.
* "In this [paper](https://arxiv.org/abs/1706.08928) we cluster 330 classical music pieces collected from MusicNet database based on their musical note sequence. We use shingling and chord trajectory matrices to create signature for each music piece and performed spectral clustering to find the clusters. Based on different resolution, the output clusters distinctively indicate composition from different classical music era and different composing style of the musicians."
* After the phenomenal success of [neural style transfer](https://www.youtube.com/watch?v=R39tWYYKNcI), now here comes [musical translation](https://www.youtube.com/watch?v=vdxCqNWTpUs&feature=youtu.be). Translating between different instruments / composers. Really neat work. You can read the [paper](https://arxiv.org/abs/1805.07848] too.

### Medical
* A fantastic [write-up](http://juliandewit.github.io/kaggle-ndsb2017/) on a solution to lung cancer detection from 3D CT scans. Inference from 3D systems are significantly more complicated compared to 2D (say images), so this is a very interesting write-up. Also it covers a lot of fundamentals on machine learning, such as handling data (preprocessing), model selection, along with the code itself.
* [Learn](https://research.googleblog.com/2018/02/assessing-cardiovascular-risk-factors.html) how deep learning makes it possible to accurately assess cardiovascular risk factors using retinal images

### Others
* A very nice [video](https://www.youtube.com/watch?v=iOWamCtnwTc&app=desktop) on how a neural net learns fluid dynamics.
* A ML [model](https://arxiv.org/abs/1705.07962) to convert from a GUI (an image) to code :). AI is entering front end stack space and like always, is going to disrupt humans and existing companies.
* If you've always wondered how Google keeps maps updated, the answer is a lot of manual inputs via "Google local guides" and in many places Deep learning clubbed with street view. An [article](https://research.googleblog.com/2017/05/updating-google-maps-with-deep-learning.html) discussing the challenges along with their solution. Wonderful to read solutions to real life problems.
* Going about solving [social problems](https://hbr.org/2016/12/a-guide-to-solving-social-problems-with-machine-learning
) using machine learning from Harvard.
* An AI that uses [neural networks](https://blog.acolyer.org/2017/03/29/deepcoder-learning-to-write-programs/) to write programs and solve programming challenges. This is a fantastic step forward because it's the start of programs being able to write out simpler programs
* [Astronomers](http://www.nature.com/news/astronomers-explore-uses-for-ai-generated-images-1.21398) using AI generated images (using deep learning):
* Microsoft has taken a unique approach of using [NLP to predict stock prices](https://www.microsoft.com/developerblog/2017/12/04/predicting-stock-performance-deep-learning/). 
