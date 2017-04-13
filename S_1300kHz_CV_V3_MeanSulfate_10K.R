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

frequency_1300 <-rep(1300, 25) # kHz
power_1300 <- c(51,	90,	137,	181,	223,	51,	90,	137,	181,	223,	51,	90,	137,	181,	223,	51,	90,	137,	181,	223,	51,	90,	137,	181,	223)
surfate_conc_1300 <- c(0.1011,	0.0703,	0.049,	0.0568,	0.014,	0.051,	0.1268,	0.0991,	0.0122,	0.1007,	0.1572,	0.0532,	0.08,	0.0483,	0.1028,	0.0348,	0.0901,	0.028,	0.1121,	0.0307,	0.0604,	0.1178,	0.0499,	0.1232,	0.0362)
# Put all reqindividual data in a table
full_data <- data.frame(frequency = frequency_1300, temperature = temperature, power = power_1300, air = air, Ar = Ar, Ar_O2 = Ar_O2, He = He, O2 = O2, surfate_conc = surfate_conc_1300)

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

net.data <- neuralnet(surfate_conc ~ frequency + temperature + power + air + Ar + Ar_O2 + He + O2, full_data, 
                      hidden = best_neuron, startweights = best_startweights, 
                      err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = best_learning_rate)

net.data$net.result
net.data$data$surfate_conc
net.data$result.matrix[1,]
predict.data <- neuralnet::compute(net.data, full_data[,1:8])
predict.data$net.result
df <- data.frame(net.data$net.result, net.data$response)
scaled_df <- data.frame(df[,1] * 10,df[,2] * 10) # Scaling back to the original 
names(scaled_df)[1]<- paste("predicted")
names(scaled_df)[2]<- paste("desired")
scaled_df


########################################################## Calculating mean_sulfate###########################
## Temperature
temp_5 <- mean(scaled_df[1:5, 1])
temp_20 <- mean(scaled_df[6:10, 1])
temp_35 <- mean(scaled_df[11:15, 1])
temp_50 <- mean(scaled_df[16:20, 1])
temp_10 <- mean(scaled_df[21:25, 1])
temp_1300_meanS <- data.frame(c(5, 10, 20, 35, 50), c(temp_5, temp_10, temp_20, temp_35, temp_50))
names(temp_1300_meanS)[1]<- paste("Temperature (C)")
names(temp_1300_meanS)[2]<- paste("mean Sulfate (mM)")
temp_1300_meanS 

## Power
power_13 <- mean(scaled_df[1, 1], scaled_df[6, 1], scaled_df[11, 1], scaled_df[16, 1], scaled_df[21, 1])
power_22 <- mean(scaled_df[2, 1], scaled_df[7, 1], scaled_df[12, 1], scaled_df[17, 1], scaled_df[22, 1])
power_33 <- mean(scaled_df[3, 1], scaled_df[8, 1], scaled_df[13, 1], scaled_df[18, 1], scaled_df[23, 1])
power_43 <- mean(scaled_df[4, 1], scaled_df[9, 1], scaled_df[14, 1], scaled_df[19, 1], scaled_df[24, 1])
power_55 <- mean(scaled_df[5, 1], scaled_df[10, 1], scaled_df[15, 1], scaled_df[20, 1], scaled_df[25, 1])
power_1300_meanS <- data.frame(c(13, 22, 33, 43, 55), c(power_13, power_22, power_33, power_43, power_55))
names(power_1300_meanS)[1]<- paste("Power (W)")
names(power_1300_meanS)[2]<- paste("mean Sulfate (mM)")
power_1300_meanS

## Gas
gas_He <- mean(scaled_df[1, 1], scaled_df[10, 1], scaled_df[14, 1], scaled_df[18, 1], scaled_df[22, 1])
gas_Air <- mean(scaled_df[2, 1], scaled_df[6, 1], scaled_df[15, 1], scaled_df[19, 1], scaled_df[23, 1])
gas_O2 <- mean(scaled_df[3, 1], scaled_df[7, 1], scaled_df[11, 1], scaled_df[20, 1], scaled_df[24, 1])
gas_Ar <- mean(scaled_df[4, 1], scaled_df[8, 1], scaled_df[12, 1], scaled_df[16, 1], scaled_df[25, 1])
gas_Ar_O2 <- mean(scaled_df[5, 1], scaled_df[9, 1], scaled_df[13, 1], scaled_df[17, 1], scaled_df[21, 1])
gas_1300_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_He, gas_Air, gas_O2, gas_Ar, gas_Ar_O2))
names(gas_1300_meanS)[1]<- paste("Gas")
names(gas_1300_meanS)[2]<- paste("mean Sulfate (mM)")
gas_1300_meanS
