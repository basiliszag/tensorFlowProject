##############################################################################################################################
###Amazon Product Reviews - Sentiment Analysis using RNN project                                                           ### 
##############################################################################################################################
###In this notebook, we have implement a RNN that performs polarity sentiment analysis.
###The dataset used for input is coming from Kaggle- https://www.kaggle.com/snap/amazon-fine-food-reviews 
###and consists of 568,454 reviews. You may need to manage the csv externally before importing in Python - 
###there is a special character that can not be loaded.                                                           
###Team members of the project are : Giota Koufonikou, Zaglaras Vasilis, Konstantinos Lolos, George Pierrakos, Sofia Kouki
###The code is splitted to 4 main parts :Data Preparation, Building the Graph, Training and Validation and Testing. 
###You can execute in sequence the different parts,uncomment some lines to take printouts.
###You can change some constants for dataset selection and model hyper-parameters
##############################################################################################################################

#####################################################################################################
#####################   Import Libraries    #########################################################
#####################################################################################################
import numpy as np
import pandas as pd
import nltk

import tensorflow as tf

#####################################################################################################
#####################   Prepare Data set to feed the model      #####################################
#####################################################################################################

##Path to Reviews File
path_in = '/DataInput/AmazonReviews/Reviews_new.csv'

##Read file using pandas - creates a data frame
#create data frame using read_csv from pandas
Reviews = pd.read_csv(path_in, sep=",", header = 'infer')

##keep only score and text columns
Reviews.drop(Reviews.columns[[0,1,2,3,4,5,7,8]], axis=1, inplace=True)

#print(len(Reviews)) - #568454 reviews - to many reviews to handle

##remove punctuation and then lower case in a new column
Reviews['CleanReview']=Reviews['Text'].astype(str)
Reviews['CleanReview']=Reviews['CleanReview'].str.replace('[^a-zA-Z\s]','').str.lower()

#function to return words count
def review_words(review_row):
    words = review_row.split()
    return( len(words))

#Store review words count in a separate column
Reviews['WordCount']=Reviews['CleanReview'].apply(lambda x: review_words(x))

##reviews max and min length
print("Minimum-length reviews: {}".format(min(Reviews['WordCount'])))
print("Maximum review length: {}".format(max(Reviews['WordCount'])))

#Minimum-length reviews: 0
#Maximum review length: 3393
###the maximum review length is way too many steps for our RNN. Let's truncate to less steps.
###we will use a variable to define the n_steps to filter our reviews

n_steps=100
#keep reviews with non zero max length 100 words 
ReviewsN=Reviews.loc[Reviews['WordCount'] <= n_steps]
ReviewsN=ReviewsN.loc[Reviews['WordCount'] > 0]

#print(len(ReviewsN))
#435219 reviews with max length 100 words
print(ReviewsN.groupby('Score').count())
##Group by Score(Label)
#1 - 38797 
#2 - 21130 
#3 - 28921 
#4 - 56522 
#5 - 289849

#The model will predict only negative/positive, so we will change the score 1,2 to 0 and 3,4 to 1
#Score 3 is ignored as neutral

#Random Selection of specific reviews by group
ReviewsNo = 30000  #30000 for scores 1,4,5 and 21000 for score 2 since is less than 30000
Reviews5 = ReviewsN.loc[ReviewsN['Score'] == 5].sample(ReviewsNo)
Reviews4 = ReviewsN.loc[ReviewsN['Score'] == 4].sample(ReviewsNo)
Reviews2 = ReviewsN.loc[ReviewsN['Score'] == 2].sample(21000)
Reviews1 = ReviewsN.loc[ReviewsN['Score'] == 1].sample(ReviewsNo)

ReviewsF = Reviews1.append([Reviews2, Reviews4, Reviews5])
print(len(ReviewsF))
#111000

#Random selection of total number of reviews
TotalReviewsNo = 110000
ReviewsF=ReviewsF.sample(TotalReviewsNo)

#Reviews Random group by Score
print(ReviewsF.groupby('Score').count())
#1 - 29730 
#2 - 20809 
#4 - 29735 
#5 - 29726 

#Join all the words to build a corpus
all_text = ' '.join(ReviewsF['CleanReview'])
words = all_text.split()

# Count the word frequencies
word_freq = nltk.FreqDist(words)
print ("Found %d unique words tokens." % len(word_freq.items())) 
#Found 63987  unique words tokens

#Create the dictionary that maps vocab words to integers
#Later we're going to pad our input vectors with zeros, so the integers start at 1, not 0
from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
#print(len(vocab_to_int)) #-63987

#create reviews to int using the vocabulary 
reviews_ints = []
for each in ReviewsF['CleanReview']:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])
#print(len(reviews_ints)) #- 110000


#create an array features that contains the data we'll pass to the network 
#The data should come from review_ints, since we want to feed integers to the network 
#Each row should be n_steps elements long. For reviews shorter than n_steps, left pad with 0s

seq_len = n_steps
features = np.zeros((len(reviews_ints), seq_len), dtype=int)

for i, row in enumerate(reviews_ints):
    if len(row)>0: 
        features[i, -len(row):] = np.array(row)[:seq_len]

#print test row
print(len(features))
print(type(features))
print(features[10])
print(len(features[10]))
print(reviews_ints[10])
print(len(reviews_ints[10]))

#Convert score to binary - values 1-2 means negative(0), values 4,5 means positive(1)
ReviewsF['Score_label']=ReviewsF['Score'].apply(lambda x: 0 if x < 3 else 1)
#labels set as array
labels=np.array(ReviewsF['Score_label'])

#split data set into training, validation, and test sets
#Split fraction is set to 0.9 for training.The rest of the data are split in half to create the validation and testing data.
split_frac = 0.9

split_index = int(split_frac * len(features))

train_x, val_x = features[:split_index], features[split_index:] 
train_y, val_y = labels[:split_index], labels[split_index:]

split_frac = 0.5
split_index = int(split_frac * len(val_x))

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape), 
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))

#With train, validation, and text fractions of 0.9, 0.5, 0.5, the final shapes looks like:
#Train set: (99000, 100) 
#Validation set: (5500, 100) 
#Test set: (5500, 100)
#label set:  (99000,) 
#Validation label set: (5500,) 
#Test label set: (5500,)

#Train set labels count
#np.unique(train_y,return_counts=True) #0-45447, 1-53553
#Val set labels count
#np.unique(val_y,return_counts=True)   #0-2546, 1-2954
#Test set labels count
#np.unique(test_y,return_counts=True)  #0-2546, 1-2954


#####################################################################################################
#####################   Build the graph                         #####################################
#####################################################################################################
######Define Hyper-Parameters######
#lstm_size -> Number of units in the hidden layers in the LSTM cells.Larger is better performance wise.Common values are 128, 256, 512, etc. 
#lstm_layers -> Number of LSTM layers in the network. Start with 1, then add more if underfitting
#batch_size -> The number of reviews to feed the network in one training pass. Typically this should be set as high as you can go without running out of memory.
#learning_rate -> Learning rate
lstm_size = 256
lstm_layers = 2
batch_size = 1000
learning_rate = 0.01

#For the network itself, we'll be passing in our 198(max review length) element long review vectors. 
#Each batch will be batch_size vectors. 
#We'll also be using dropout on the LSTM layer for addressing overfitting, so we'll make a placeholder for the keep probability.

n_words = len(vocab_to_int) + 1 # Add 1 for 0 added to vocab

#Create the graph object
tf.reset_default_graph()
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels") #labels_ needs to be two-dimensional to work with some functions later
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") #keep_prob is a scalar (a 0-dimensional tensor), we shouldn't provide a size
    
######Embedding######
#We need to add an embedding layer because there are 10522 words in our vocabulary. 
#It is massively inefficient to one-hot encode our classes here. 
#Instead of one-hot encoding, we can have an embedding layer and use that layer as a lookup table and let the network learn the weights

embed_size = 300 # Size of the embedding vectors (number of units in the embedding layer)

with tf.name_scope("Embeddings"):
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

######LSTM cell######
#Define what the cells look like
#To create a basic LSTM cell for the graph, we use the function tf.contrib.rnn.BasicLSTMCell with num_units=lstm_size and forget_bias=1.0(default)
#We add dropout to the cell with tf.contrib.rnn.DropoutWrapper function. This just wraps the cell in another cell, but with dropout added to the inputs and/or outputs.
#The network will have better performance with more layers. Adding more layers allows the network to learn really complex relationships. 
#To create multiple layers of LSTM cells we use tf.contrib.rnn.MultiRNNCell function 

def lstm_cell():
    # Our basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    # Add dropout to the cell
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

with tf.name_scope("RNN_layers"):
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
    
    #[drop] * lstm_layers creates a list of cells (drop) that is lstm_layers long 
    #The MultiRNNCell wrapper builds this into multiple layers of RNN cells, one for each cell in the list
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)   
    
######RNN forward pass######
#We actually run the data through the RNN nodes using tf.nn.dynamic_rnn 
#by passing the inputs(vectors from the embedding layer) to the network to the multiple layered LSTM cell    

with tf.name_scope("RNN_forward"):
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

#We created an initial state to pass to the RNN. This is the cell state that is passed between the hidden layers in successive time steps.
#It returns outputs for each time step and the final_state of the hidden layer.    

######Output######
#We only care about the final output, we'll be using that as our sentiment prediction. 
#So we need to grab the last output with outputs[:, -1], the cost from that and labels_.

with tf.name_scope('predictions'):
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels_, predictions)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

######Validation accuracy######
#We add a few nodes to calculate the accuracy which we'll use in the validation pass
with tf.name_scope('validation'):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
######Batching######
#This is a simple function for returning batches from our data. 
#First it removes data such that we only have full batches. 
#Then it iterates through the x and y arrays and returns slices out of those arrays with size [batch_size].

def get_batches(x, y, batch_size):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        
#####################################################################################################
#####################   Training and validation in batches      #####################################
#####################################################################################################

###Training and validation in batches
###Once the graph is defined, training can be done in batches based on the batch_size hyper parameter.        

n_epochs = 10
#batches = len(train_x)//batch_size
display_step = 1

# with graph.as_default():
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #Initialize all variables
    #matrices to store values for charts
    epochs = []  #Iterations
    costs = []  #Loss
    tr_acc = []  #Training Accuracy
    val_acc = [] #Val Accuracy
    b_cost= [] #Training Bach Cost
    b_tr_acc = [] #Training Bach Accuracy
    b_te_acc = [] #Val Bach Accuracy
       

    for e in range(n_epochs):
        state = sess.run(initial_state)
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}

            batch_acc, loss, state, _ = sess.run([accuracy, cost, final_state, optimizer], feed_dict=feed)
            b_tr_acc.append(batch_acc)
            b_cost.append(loss)
            
        if (e+1) % display_step == 0:
        #calculate val acc
            val_state = sess.run(cell.zero_state(batch_size, tf.float32))
            for x, y in get_batches(val_x, val_y, batch_size):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 1,
                        initial_state: val_state}
                batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                b_te_acc.append(batch_acc)

            print("Epoch: "+str('%04d' % (e+1))+" Cost="+"{:.9f}".format(np.mean(b_cost))+
                  " Train accuracy="+"{:.9f}".format(np.mean(b_tr_acc))+
                  " Val accuracy="+"{:.9f}".format(np.mean(b_te_acc)))
            # add results to matrices at the end of each epoch
            epochs.append(e+1)
            tr_acc.append(np.mean(b_tr_acc))
            val_acc.append(np.mean(b_te_acc))
            costs.append(np.mean(b_cost))
        
        saver.save(sess, './sentiment_model.ckpt')
  
    print("Optimization Finished!")
    
#####################################################################################################
##################### Charts for learning curve and train cost  #####################################
#####################################################################################################

%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (20, 6)
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

fig1 = plt.figure()

ax = fig1.add_subplot(121)
ax.clear()

ax.set_title("Learning curve")
ax.set_xlabel('Iterations')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.plot(epochs, tr_acc,     'o-', color="g", label="Train Accuracy")
ax.plot(epochs, val_acc,   'o-', color="r", label="Test Accuracy")

ax2 = fig1.add_subplot(122)
ax2.clear()

ax2.set_title("Train cost")
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_ylim(ymin=0)
ax2.plot(epochs, costs, 'o-', color="r", label="Train cost")


#####################################################################################################
#####################   Testing             #########################################################
#####################################################################################################    
    
test_pred = []
test_acc = []
with tf.Session() as sess:
    saver.restore(sess, './sentiment_model.ckpt')
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
        
        prediction = tf.cast(tf.round(predictions),tf.int32)
        prediction = sess.run(prediction,feed_dict=feed)
        test_pred.append(prediction)


#####################################################################################################
#####################   Results             #########################################################
#####################################################################################################    

##############Confusion Matrix######################    
test_pred_flat = (np.array(test_pred)).flatten()
y_act = pd.Series(test_y, name='Actual')
y_pred = pd.Series(test_pred_flat, name='Predicted')
df_confusion = pd.crosstab(y_act, y_pred, margins=True)

print("Test accuracy: {:.3f}".format(np.mean(test_acc)))  
print("Confusion Matrix")
print("----------------")
print(df_confusion)
    

##############Review Text for Testing Results######################    
#Take Reviews for test set
start_idx = len(train_x) + len(val_x)
end_idx = start_idx + len(test_pred_flat)
ReviewsTest=ReviewsF.iloc[start_idx:end_idx]
#Add Predicted Sentiment in a new column
ReviewsTest['Predicted_Sentiment']= test_pred_flat

#Examples of False Negative Results
ReviewsTest[(ReviewsTest['Score_label'] == 1) & (ReviewsTest['Predicted_Sentiment'] == 0 )].iloc[1:10]    
#Examples of False Positive Results
ReviewsTest[(ReviewsTest['Score_label'] == 0) & (ReviewsTest['Predicted_Sentiment'] == 1 )].iloc[1:10]    

##############WordCloud for Positive and Negative Sentiment######################
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
##Reviews with sentiment as Positive
posActualReviews = ReviewsTest[ReviewsTest.Score_label==1]
posPredReviews = ReviewsTest[ReviewsTest.Predicted_Sentiment==1]

fig = plt.figure( figsize=(40,40))
##Generate a word cloud image for Actual Positive Sentiment
sub1= fig.add_subplot(2,2,1)
plt.title("Positive Sentiment - Actual")
ReviewText = ' '.join((posActualReviews['CleanReview']))
wordcloud = WordCloud().generate(ReviewText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
 
##Generate a word cloud image for Predicted Positive Sentiment
sub2= fig.add_subplot(2,2,2)
plt.title("Positive Sentiment - Prediction")
ReviewText = ' '.join((posPredReviews['CleanReview']))
wordcloud = WordCloud().generate(ReviewText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

##Reviews with sentiment as Negative
negActualReviews = ReviewsTest[ReviewsTest.Score_label==0]
negPredReviews = ReviewsTest[ReviewsTest.Predicted_Sentiment==0]

fig = plt.figure( figsize=(40,40))
##Generate a word cloud image for Actual Positive Sentiment
sub1= fig.add_subplot(2,2,1)
plt.title("Negative Sentiment - Actual")
ReviewText = ' '.join((negActualReviews['CleanReview']))
wordcloud = WordCloud().generate(ReviewText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
 
##Generate a word cloud image for Predicted Negative Sentiment
sub2= fig.add_subplot(2,2,2)
plt.title("Negative Sentiment - Prediction")
ReviewText = ' '.join((negPredReviews['CleanReview']))
wordcloud = WordCloud().generate(ReviewText)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
