## Loads neuralnet package
library('dplyr')
library('neuralnet')
library('mlr')
library('caret')

######################################### RAW DATA ########################################

temperature <- c(5,	5,	5,	5,	5,	20,	20,	20,	20,	20,	35,	35,	35,	35,	35, 50,	50,	50,	50,	50,	10,	10,	10,	10,	10) # [=] C
gastype <- data.frame(var = c("helium", "air", "O2", "Ar", "Ar/O2", "air", "O2", "Ar", "Ar/O2", "helium", "O2", "Ar", "Ar/O2", "helium", "air", "Ar", "Ar/O2", "helium", "air", "O2", "Ar/O2", "helium", "air", "O2", "Ar"))
gas <- createDummyFeatures(gastype, cols = "var") # Creates a dummy variable for categorical types

# Individualize each gas types
air <-(gas[,1])
Ar <- (gas[,2])
Ar_O2 <- (gas[,3])
He <- (gas[,4])
O2 <- (gas[,5])

frequency_323 <-rep(323, 25) # [=] kHz
power_323 <- c(13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55) # [=] W
surfate_conc_323 <- c(0.1846,	0.1131,	0.1288,	0.0571,	0.0105,	0.0538,	0.0565,	0.1646,	0.0135,	0.0807,	0.1238,	0.0481,	0.0878,	0.0564,	0.0525,	0.1247,	0.0891,	0.1257,	0.0523,	0.0565,	0.0854,	0.0521,	0.1157,	0.0505,	0.0986) # [=] mM
# Put all reqindividual data in a table
full_data <- data.frame(frequency = frequency_323, temperature = temperature, power = power_323, air = air, Ar = Ar, Ar_O2 = Ar_O2, He = He, O2 = O2, surfate_conc = surfate_conc_323)

############################### RUNNING NEURAL NETWORK ###########################################
# 
# min_error <- 1
# learning_rate <- c(0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0001)
# 
# for (lr_i in 1:length(learning_rate)) {
#   curr_learningRate = learning_rate[lr_i]
#   for (neuron in 1:100) {
#     net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, 
#                           full_data, hidden = neuron, err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, 
#                           learningrate = curr_learningRate)
#     
#     error <- net.data$result.matrix[1,]
#     
#     if (min_error > error) {
#       min_error <- error
#       best_neuron <- neuron
#       best_learning_rate <- curr_learningRate
#       best_startweights <- net.data$startweights
#     }
#   }
# }
# min_error
# best_neuron
# best_learning_rate

############################### K-FOLD CROSS VALIDATION ######################################

k <- 10
## 90% of the sample size
training_sample_size <- floor(0.9 * nrow(full_data)) ## 90% of training set

## set the seed to make your partition reproductible
set.seed(1)
## This randomly shuffles the data
cv_error_array <- NULL
shuffled_data <- full_data[sample(nrow(full_data)),]

#Create k-equally sized folds
folds <- cut(seq(1,nrow(shuffled_data)),breaks=k,labels=FALSE)

learning_rate <- c(0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0001)
min_cv_error <- 1

for (lr_i in 1:length(learning_rate)) {
  curr_learningRate = learning_rate[lr_i]
  for (neuron in 1:100) {
    for(i in 1:k) {
    #Segement your data by fold using the which() function 
      testIndexes <- which(folds==i,arr.ind=TRUE)
      test_data <- shuffled_data[testIndexes, ]
      train_data <- shuffled_data[-testIndexes, ]
      if (i == 1) {
        net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, train_data, 
                              hidden = neuron, err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = curr_learningRate)
        curr_startweights <- net.data$startweights
      }
      else {
        net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, train_data, 
                              hidden = neuron, startweights = curr_startweights, err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = curr_learningRate)
      }
      
      predict.data <- neuralnet::compute(net.data, test_data[,1:8]) ## Predicts the surfate_conc with the test_data parameters
      cv_error_array[i] <- sum((predict.data$net.result - test_data[,9])^2)/nrow(test_data)
    }
    
    cv_error <- mean(cv_error_array)
    
    if (min_cv_error > cv_error) {
      min_cv_error <- cv_error
      best_neuron <- neuron
      best_learning_rate <- curr_learningRate
      best_startweights <- curr_startweights
    }
  }
}

min_cv_error
best_neuron
best_learning_rate
#best_startweights

net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, full_data, 
                      hidden = best_neuron, startweights = best_startweights, 
                      err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = best_learning_rate)

net.data$net.result
net.data$data$surfate_conc
net.data$result.matrix[1,]
predict.data <- neuralnet::compute(net.data, full_data[,1:8])
predict.data$net.result
df <- data.frame(net.data$net.result, net.data$response)
corrected_df <- data.frame(df[,1],df[,2]) # Scaling back to the original 
names(corrected_df)[1]<- paste("predicted")
names(corrected_df)[2]<- paste("desired")
corrected_df


# ########################################################## Calculating mean_sulfate for temperature ###########################
# 
# 
# 
# 
# 
# ##################################################################################################################
# # Calculate mean error for cross fold validation
# 
# cv_error <- mean(cv_error_array)
# cv_error
# best_neuron #62 71 99 92
# min_error#
# 
# df <- data.frame(net.data$net.result, net.data$response)
# corrected_df <- data.frame(df[,1]/100,df[,2]/100) # Scaling back to the original 
# names(corrected_df)[1]<- paste("predicted")
# names(corrected_df)[2]<- paste("desired")
# 
# cat("The optimal number of neurons for 323kHz is: ", best_neuron)
# cat("Minimum error with sum squared error method is: ", min_error)
# print("The final result table is as follows: ")
# 
# corrected_df
# 
# plot(net.data)
