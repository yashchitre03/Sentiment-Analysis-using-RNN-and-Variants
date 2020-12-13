# Sentiment analysis using RNN and variants
In this project, I have trained and evaluated RNN, LSTM and GRU models,
on the task of predicting the sentiment of a text on the Rotten Tomatoes dataset.
This dataset used to train and test the models is available at:
[Movie review dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) under the “Sentiment polarity datasets” section and precisely the dataset under the ‘sentence
polarity dataset v1.0’ link. 
The dataset contains 5,331 positive and 5,331 negative processed sentences /
snippets.
Word-level representations for the models were used, specifically, the “glove.6B.zip” version trained on “Wikipedia 2014 + Gigaword 5” and 300d word vectors.
The glove word embeddings are available from [Glove word embeddings](https://nlp.stanford.edu/projects/glove/).

## Models tested
1. Vanilla RNN model
2. LSTM model
3. GRU model

## Training runs
* The model used is sequential model that has mask_zero set to True in order to get the hidden state corresponding to the last word in the sentence (ignoring padding).
* Also, I have set an early stopping criteria, so if the validation loss does not improve after at max 2 epochs, the training is halted right there. Hence, even though the epoch limit is set to 15, some training runs stop way before that.
* The vanilla RNN model has been run on the training dataset for a total of **9 times** using different hyper-parameters.
* First run consists of the one with the most optimal parameters.
* Result for each of the run is discussed accordingly.
* After finding the optimal parameters for the vanilla RNN, the same hyper-paramters have been used for the LSTM and GRU models.
* Finally, I have passed the testing data to the model of each variant. Results for testing are discussed at the end.

1. Vanilla RNN (most optimal):
    * This run uses the right balance of the batch size, number of hidden neurons, sequence length, and learning rate.
    * We will observe later on what happens when these parameters are changed, and how they affect the training.
    * For the vanilla RNN these parameters leads to a good 70% plus validation accuracy.

2. Vanilla RNN (smaller batch size):
    * Here the batch size is significantly decreased, from 100 to just 10.
    * We can see that this model also achieves average results on the validation set, sometimes slightly closer to the ideal run.
    * But, since the batch size is small each epoch takes a lot of time to run (6-7 seconds compared to 1-2 seconds seen previously).
    * It is because here the model has to udpate the weights more number of times in each epoch, requiring more time.
    * This is undesireable since we have similar result but with much slower training.

3. Vanilla RNN (larger batch size):
    * Here, we get slightly lower validation accuracies than the ideal run.
    * Larger batch size means weights are updated less frequently. So even if each epoch runs faster, it is not preferable to set the batch size too large.
    * Hence, we set the batch size somewhere in the middle (100). The result can be seen in run 1.

4. Vanilla RNN (less number of neurons):
    * We have set the number of neurons to 30.
    * The number of neurons are low, so the model is unable to learn a lot from the data.
    * This can be seen in the overall lower (peak) accuracy both on the training and the validation dataset.
    * We need to increase the hidden neurons so that the model can learn more and generalize from the training dataset.

5. Vanilla RNN (more number of neurons):
    * This model learns relatively well from the training data.
    * But it is still not desireable as the model starts over-fitting on the training data.
    * We can observe the gap between training loss and validation loss is a lot wider.
    * So, I finally settled on around 75 neurons for the optimal parameters.

6. Vanilla RNN (shorter sequence length):
    * In the ideal run (1), we set the sequence length as the longest sentence in our dataset.
    * Even after halving the sequence, we can still observe good results on the validation dataset.
    * Even if the accuracies are not equal to or greater than the first run, the results are surprising as we are cutting away almost half the information in the sequence for most samples.

7. Vanilla RNN (longer sequence length):
    * Since, we are ignoring the padding for our model, the results here are similar to the first run.
    * This can also be seen in the number of trainable parameters.
    * Hence, there does't seem to be much of a point in increasing the sequence length of the data.

8. Vanilla RNN (bigger learning rate):
    * As can be seen, the loss keeps going down and then up, indicating that the model might be over-shooting the global optimal point.
    * Hence, the training seems instable and the loss difference between training and validation keeps getting wider.
    * Inshort, it is better to not keep the learning rate too high.

9. Vanilla RNN (smaller learning rate):
    * When the learning rate is too small, the weight updates are also too small, so the model doesn't make much progress on the loss and accuracies.
    * Even after 15 epochs, the loss is still relatively high. We would have to train for many more epochs to reach near our ideal accuracies.
    * So keeping the learning rate too small is never recommended.

10. LSTM (with RNN's optimal parameters):
    * Even after using the same parameters as from our RNN model, we can see significant bump in the model accuracy on the validation set.
    * But the number of trainable parameters for this model is significantly higher than the vanilla RNN model.
    * This is due to the behavior of LSTM to remember long term dependencies in the data samples.

11. GRU (with RNN's optimal parameters):
    * Similar to the LSTM model, here too we can see significant bump in the model's accuracy on the validation set.
    * GRU is sort of a simplified version of the LSTM, even though it may not have dedicated memory units, it still manages to learn a lot of dependencies.
    * Number of trainable parameters are higher than vanilla RNN, but lower than LSTM.

## Analysis for all models:
* Looking at the test data results, overall all three models were able to make pretty significant improvements after running for at maximum 15 number of epochs.
* But we have some interesting differences to note between them.
* The LSTM and GRU models managed to get more accurate overall on the training, validation as well as the test set. This can also be seen in the F1-score (which is dependent on the precision and recall) of the three models.
* This could be attributed to the ability of these networks to learn long term information from the data sequence.
* But that does't mean they are clear winners for this dataset. Due to the inherent architecture of LSTM and GRU models, they are more complex and have more parameters to train.
* For vanilla RNN, we have approximately 28k trainable parameters (run 1 of RNN), but for LSTM, we have around 112k trainable parameters and 84k for GRU.
* Hence, vanilla RNN is not only easier to train, but will also use less memory. Where vanilla RNN takes around 2 seconds, the others take around 6 seconds for each epoch of training on my machine.
* So, if I had to choose between the three, I would choose the GRU network, as it not only has better results (sometimes even better than LSTM) in terms of accuracies and the F-1 score, but also uses significantly less number of parameters than the LSTM.
* Choosing GRU, we get best of both worlds, good results as well as low resource use (compared to LSTM) while training and storing the model.
* I feel LSTM would be more appropriate for longer and complex data sequences, where the memory units would really shine in storing long term dependencies (advantageous over the vanishing gradient problem of the vanilla RNN). Though GRU is sufficient for the current dataset.
