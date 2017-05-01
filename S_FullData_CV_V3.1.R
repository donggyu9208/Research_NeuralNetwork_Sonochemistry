## Loads neuralnet package
library('dplyr')
library('neuralnet')
library('mlr')
library('caret')
library('ggplot2')
library('ggalt')
library('reshape2')


######################################### RAW DATA ########################################

temperature <- c(5,	5,	5,	5,	5,	20,	20,	20,	20,	20,	35,	35,	35,	35,	35, 50,	50,	50,	50,	50,	10,	10,	10,	10,	10) # [=] C
temperature_total <- rep(temperature, 4);
gas = c("helium", "air", "O2", "Ar", "Ar/O2", "air", "O2", "Ar", "Ar/O2", "helium", "O2", "Ar", "Ar/O2", "helium", "air", "Ar", "Ar/O2", "helium", "air", "O2", "Ar/O2", "helium", "air", "O2", "Ar")
gastype <- data.frame(gas = rep(gas, 4))

# Individualize each gas types
gas_total <- createDummyFeatures(gastype, cols = 'gas')
air_total <-(gas_total[,1])
Ar_total <- (gas_total[,2])
Ar_O2_total <- (gas_total[,3])
He_total <- (gas_total[,4])
O2_total <- (gas_total[,5])

frequency_323 <-rep(323, 25) # kHz
power_323 <- c(13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55,	13,	22,	33,	43,	55)
surfate_conc_323 <- c(0.1846,	0.1131,	0.1288,	0.0571,	0.0105,	0.0538,	0.0565,	0.1646,	0.0135,	0.0807,	0.1238,	0.0481,	0.0878,	0.0564,	0.0525,	0.1247,	0.0891,	0.1257,	0.0523,	0.0565,	0.0854,	0.0521,	0.1157,	0.0505,	0.0986)

frequency_581 <-rep(581, 25) # kHz
power_581 <- c(49,	81,	117,	153,	185,	49,	81,	117,	153,	185,	49,	81,	117,	153,	185,	49,	81,	117,	153,	185,	49,	81,	117,	153,	185)
surfate_conc_581 <- c(0.059,	0.0548,	0.1327,	0.064,	0.0193,	0.0341,	0.1289,	0.0484,	0.0022,	0.1621,	0.1028,	0.1158,	0.0695,	0.0355	,0.1012	,0.1597,	0.0161	,0.1028	,0.0354,	0.06,	0.0925,	0.0519	,0.0583	,0.1166,	0.0584)

frequency_611 <-rep(611, 25) # kHz
power_611 <- c(39,	68,	104,	139,	175,	39,	68,	104,	139,	175,	39, 68,	104,	139,	175,	39,	68,	104,	139,	175,	39,	68,	104,	139	,175)
surfate_conc_611 <- c(0.1247,	0.0566,	0.129,	0.0575,	0.0038,	0.1461,	0.0592,	0.0583,	0.0162,	0.093,	0.1356,	0.0597,	0.0752,	0.1251,	0.0703	,0.1338,	0.0162,	0.0642,	0.0525,	0.0529,	0.0785,	0.0826,	0.1255,	0.0564,	0.0534)

frequency_1300 <-rep(1300, 25) # kHz
power_1300 <- c(51,	90,	137,	181,	223,	51,	90,	137,	181,	223,	51,	90,	137,	181,	223,	51,	90,	137,	181,	223,	51,	90,	137,	181,	223)
surfate_conc_1300 <- c(0.1011,	0.0703,	0.049,	0.0568,	0.014,	0.051,	0.1268,	0.0991,	0.0122,	0.1007,	0.1572,	0.0532,	0.08,	0.0483,	0.1028,	0.0348,	0.0901,	0.028,	0.1121,	0.0307,	0.0604,	0.1178,	0.0499,	0.1232,	0.0362)

frequency_total <- c(frequency_323, frequency_581, frequency_611, frequency_1300)
power_total <- c(power_323, power_581, power_611, power_1300)
surfate_conc_total <- c(surfate_conc_323, surfate_conc_581, surfate_conc_611, surfate_conc_1300)

full_data <- data.frame(frequency = frequency_total, temperature = temperature_total, power = power_total, air = air_total, Ar = Ar_total, Ar_O2 = Ar_O2_total, He = He_total, O2 = O2_total, sulfate_conc = surfate_conc_total)

############################### K-FOLD CROSS VALIDATION WITH NEURAL NETWORK (K = 10) ######################################
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

learning_rate <- c(0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001)
min_cv_error <- 1

for (lr_i in 1:length(learning_rate)) {
  curr_learningRate = learning_rate[lr_i]
  for (neuron in 1:100) {
    for(i in 1:k) {
      #Segement your data by fold using the which() function 
      testIndexes <- which(folds==i,arr.ind=TRUE)
      test_data <- shuffled_data[testIndexes, ]
      train_data <- shuffled_data[-testIndexes, ]
      net.data <- neuralnet(surfate_conc_total ~ frequency_total + temperature_total + power_total + air_total + Ar_total + Ar_O2_total + He_total + O2_total, train_data, 
                              hidden = neuron, err.fct = 'sse', act.fct = 'logistic', linear.output = FALSE, learningrate = curr_learningRate)
      
      predict.data <- neuralnet::compute(net.data, test_data[, 1:8]) ## Predicts the surfate_conc with the test_data parameters
      cv_error_array[i] <- sum((predict.data$net.result - test_data[,9])^2)/nrow(test_data)
    }
    
    cv_error <- mean(cv_error_array)
    
    if (min_cv_error > cv_error) {
      min_cv_error <- cv_error
      best_neuron <- neuron
      best_learning_rate <- curr_learningRate
      best_startweights <- net.data$startweights
    }
  }
}

min_cv_error
best_neuron
best_learning_rate
#best_startweights

net.data <- neuralnet(surfate_conc_total ~ frequency_total + temperature_total + power_total + air_total + Ar_total + Ar_O2_total + He_total + O2_total, full_data, 
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

freq_323_data <- scaled_df[1:25, ]
freq_581_data <- scaled_df[26:50, ]
freq_611_data <- scaled_df[51:75, ]
freq_1300_data <- scaled_df[76:100, ]
########################################################## Calculating mean_sulfate###########################
## freq_323
## Temperature
temp_323_5 <- mean(freq_323_data[1:5, 1])
temp_323_20 <- mean(freq_323_data[6:10, 1])
temp_323_35 <- mean(freq_323_data[11:15, 1])
temp_323_50 <- mean(freq_323_data[16:20, 1])
temp_323_10 <- mean(freq_323_data[21:25, 1])
temp_323_meanS <- data.frame(c(5, 10, 20, 35, 50), c(temp_323_5, temp_323_10, temp_323_20, temp_323_35, temp_323_50))
names(temp_323_meanS)[1]<- paste("Temperature (C)")
names(temp_323_meanS)[2]<- paste("mean Sulfate (mM)")
temp_323_meanS 

## Power
power_323_13 <- mean(c(freq_323_data[1, 1], freq_323_data[6, 1], freq_323_data[11, 1], freq_323_data[16, 1], freq_323_data[21, 1]))
power_323_22 <- mean(c(freq_323_data[2, 1], freq_323_data[7, 1], freq_323_data[12, 1], freq_323_data[17, 1], freq_323_data[22, 1]))
power_323_33 <- mean(c(freq_323_data[3, 1], freq_323_data[8, 1], freq_323_data[13, 1], freq_323_data[18, 1], freq_323_data[23, 1]))
power_323_43 <- mean(c(freq_323_data[4, 1], freq_323_data[9, 1], freq_323_data[14, 1], freq_323_data[19, 1], freq_323_data[24, 1]))
power_323_55 <- mean(c(freq_323_data[5, 1], freq_323_data[10, 1], freq_323_data[15, 1], freq_323_data[20, 1], freq_323_data[25, 1]))
power_323_meanS <- data.frame(c(13, 22, 33, 43, 55), c(power_323_13, power_323_22, power_323_33, power_323_43, power_323_55))
names(power_323_meanS)[1]<- paste("Power (W)")
names(power_323_meanS)[2]<- paste("mean Sulfate (mM)")
power_323_meanS

## Gas
gas_323_He <- mean(c(freq_323_data[1, 1], freq_323_data[10, 1], freq_323_data[14, 1], freq_323_data[18, 1], freq_323_data[22, 1]))
gas_323_Air <- mean(c(freq_323_data[2, 1], freq_323_data[6, 1], freq_323_data[15, 1], freq_323_data[19, 1], freq_323_data[23, 1]))
gas_323_O2 <- mean(c(freq_323_data[3, 1], freq_323_data[7, 1], freq_323_data[11, 1], freq_323_data[20, 1], freq_323_data[24, 1]))
gas_323_Ar <- mean(c(freq_323_data[4, 1], freq_323_data[8, 1], freq_323_data[12, 1], freq_323_data[16, 1], freq_323_data[25, 1]))
gas_323_Ar_O2 <- mean(c(freq_323_data[5, 1], freq_323_data[9, 1], freq_323_data[13, 1], freq_323_data[17, 1], freq_323_data[21, 1]))
gas_323_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_323_He, gas_323_Air, gas_323_O2, gas_323_Ar, gas_323_Ar_O2))
names(gas_323_meanS)[1]<- paste("Gas")
names(gas_323_meanS)[2]<- paste("mean Sulfate (mM)")
gas_323_meanS

## freq_581
## Temperature
temp_581_5 <- mean(freq_581_data[1:5, 1])
temp_581_20 <- mean(freq_581_data[6:10, 1])
temp_581_35 <- mean(freq_581_data[11:15, 1])
temp_581_50 <- mean(freq_581_data[16:20, 1])
temp_581_10 <- mean(freq_581_data[21:25, 1])
temp_581_meanS <- data.frame(c(5, 10, 20, 35, 50), c(temp_581_5, temp_581_10, temp_581_20, temp_581_35, temp_581_50))
names(temp_581_meanS)[1]<- paste("Temperature (C)")
names(temp_581_meanS)[2]<- paste("mean Sulfate (mM)")
temp_581_meanS 

## Power
power_581_49 <- mean(c(freq_581_data[1, 1], freq_581_data[6, 1], freq_581_data[11, 1], freq_581_data[16, 1], freq_581_data[21, 1]))
power_581_81 <- mean(c(freq_581_data[2, 1], freq_581_data[7, 1], freq_581_data[12, 1], freq_581_data[17, 1], freq_581_data[22, 1]))
power_581_117 <- mean(c(freq_581_data[3, 1], freq_581_data[8, 1], freq_581_data[13, 1], freq_581_data[18, 1], freq_581_data[23, 1]))
power_581_153 <- mean(c(freq_581_data[4, 1], freq_581_data[9, 1], freq_581_data[14, 1], freq_581_data[19, 1], freq_581_data[24, 1]))
power_581_185 <- mean(c(freq_581_data[5, 1], freq_581_data[10, 1], freq_581_data[15, 1], freq_581_data[20, 1], freq_581_data[25, 1]))
power_581_meanS <- data.frame(c(49, 81, 117, 153, 185), c(power_581_49, power_581_81, power_581_117, power_581_153, power_581_185))
names(power_581_meanS)[1]<- paste("Power (W)")
names(power_581_meanS)[2]<- paste("mean Sulfate (mM)")

## Gas
gas_581_He <- mean(c(freq_581_data[1, 1], freq_581_data[10, 1], freq_581_data[14, 1], freq_581_data[18, 1], freq_581_data[22, 1]))
gas_581_Air <- mean(c(freq_581_data[2, 1], freq_581_data[6, 1], freq_581_data[15, 1], freq_581_data[19, 1], freq_581_data[23, 1]))
gas_581_O2 <- mean(c(freq_581_data[3, 1], freq_581_data[7, 1], freq_581_data[11, 1], freq_581_data[20, 1], freq_581_data[24, 1]))
gas_581_Ar <- mean(c(freq_581_data[4, 1], freq_581_data[8, 1], freq_581_data[12, 1], freq_581_data[16, 1], freq_581_data[25, 1]))
gas_581_Ar_O2 <- mean(c(freq_581_data[5, 1], freq_581_data[9, 1], freq_581_data[13, 1], freq_581_data[17, 1], freq_581_data[21, 1]))
gas_581_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_581_He, gas_581_Air, gas_581_O2, gas_581_Ar, gas_581_Ar_O2))
names(gas_581_meanS)[1]<- paste("Gas")
names(gas_581_meanS)[2]<- paste("mean Sulfate (mM)")

## freq_611
## Temperature
temp_611_5 <- mean(freq_611_data[1:5, 1])
temp_611_20 <- mean(freq_611_data[6:10, 1])
temp_611_35 <- mean(freq_611_data[11:15, 1])
temp_611_50 <- mean(freq_611_data[16:20, 1])
temp_611_10 <- mean(freq_611_data[21:25, 1])
temp_611_meanS <- data.frame(c(5, 10, 20, 35, 50), c(temp_611_5, temp_611_10, temp_611_20, temp_611_35, temp_611_50))
names(temp_611_meanS)[1]<- paste("Temperature (C)")
names(temp_611_meanS)[2]<- paste("mean Sulfate (mM)")

## Power
power_611_39 <- mean(c(freq_611_data[1, 1], freq_611_data[6, 1], freq_611_data[11, 1], freq_611_data[16, 1], freq_611_data[21, 1]))
power_611_68 <- mean(c(freq_611_data[2, 1], freq_611_data[7, 1], freq_611_data[12, 1], freq_611_data[17, 1], freq_611_data[22, 1]))
power_611_104 <- mean(c(freq_611_data[3, 1], freq_611_data[8, 1], freq_611_data[13, 1], freq_611_data[18, 1], freq_611_data[23, 1]))
power_611_139 <- mean(c(freq_611_data[4, 1], freq_611_data[9, 1], freq_611_data[14, 1], freq_611_data[19, 1], freq_611_data[24, 1]))
power_611_175 <- mean(c(freq_611_data[5, 1], freq_611_data[10, 1], freq_611_data[15, 1], freq_611_data[20, 1], freq_611_data[25, 1]))
power_611_meanS <- data.frame(c(39, 68, 104, 139, 175), c(power_611_39, power_611_68, power_611_104, power_611_139, power_611_175))
names(power_611_meanS)[1]<- paste("Power (W)")
names(power_611_meanS)[2]<- paste("mean Sulfate (mM)")

## Gas
gas_611_He <- mean(c(freq_611_data[1, 1], freq_611_data[10, 1], freq_611_data[14, 1], freq_611_data[18, 1], freq_611_data[22, 1]))
gas_611_Air <- mean(c(freq_611_data[2, 1], freq_611_data[6, 1], freq_611_data[15, 1], freq_611_data[19, 1], freq_611_data[23, 1]))
gas_611_O2 <- mean(c(freq_611_data[3, 1], freq_611_data[7, 1], freq_611_data[11, 1], freq_611_data[20, 1], freq_611_data[24, 1]))
gas_611_Ar <- mean(c(freq_611_data[4, 1], freq_611_data[8, 1], freq_611_data[12, 1], freq_611_data[16, 1], freq_611_data[25, 1]))
gas_611_Ar_O2 <- mean(c(freq_611_data[5, 1], freq_611_data[9, 1], freq_611_data[13, 1], freq_611_data[17, 1], freq_611_data[21, 1]))
gas_611_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_611_He, gas_611_Air, gas_611_O2, gas_611_Ar, gas_611_Ar_O2))
names(gas_611_meanS)[1]<- paste("Gas")
names(gas_611_meanS)[2]<- paste("mean Sulfate (mM)")

## freq_1300
## Temperature
temp_1300_5 <- mean(freq_1300_data[1:5, 1])
temp_1300_20 <- mean(freq_1300_data[6:10, 1])
temp_1300_35 <- mean(freq_1300_data[11:15, 1])
temp_1300_50 <- mean(freq_1300_data[16:20, 1])
temp_1300_10 <- mean(freq_1300_data[21:25, 1])
temp_1300_meanS <- data.frame(c(5, 10, 20, 35, 50), c(temp_1300_5, temp_1300_10, temp_1300_20, temp_1300_35, temp_1300_50))
names(temp_1300_meanS)[1]<- paste("Temperature (C)")
names(temp_1300_meanS)[2]<- paste("mean Sulfate (mM)")

## Power
power_1300_51 <- mean(c(freq_1300_data[1, 1], freq_1300_data[6, 1], freq_1300_data[11, 1], freq_1300_data[16, 1], freq_1300_data[21, 1]))
power_1300_90 <- mean(c(freq_1300_data[2, 1], freq_1300_data[7, 1], freq_1300_data[12, 1], freq_1300_data[17, 1], freq_1300_data[22, 1]))
power_1300_137 <- mean(c(freq_1300_data[3, 1], freq_1300_data[8, 1], freq_1300_data[13, 1], freq_1300_data[18, 1], freq_1300_data[23, 1]))
power_1300_181 <- mean(c(freq_1300_data[4, 1], freq_1300_data[9, 1], freq_1300_data[14, 1], freq_1300_data[19, 1], freq_1300_data[24, 1]))
power_1300_223 <- mean(c(freq_1300_data[5, 1], freq_1300_data[10, 1], freq_1300_data[15, 1], freq_1300_data[20, 1], freq_1300_data[25, 1]))
power_1300_meanS <- data.frame(c(51, 90, 137, 181, 223), c(power_1300_51, power_1300_90, power_1300_137, power_1300_181, power_1300_223))
names(power_1300_meanS)[1]<- paste("Power (W)")
names(power_1300_meanS)[2]<- paste("mean Sulfate (mM)")

## Gas
gas_1300_He <- mean(c(freq_1300_data[1, 1], freq_1300_data[10, 1], freq_1300_data[14, 1], freq_1300_data[18, 1], freq_1300_data[22, 1]))
gas_1300_Air <- mean(c(freq_1300_data[2, 1], freq_1300_data[6, 1], freq_1300_data[15, 1], freq_1300_data[19, 1], freq_1300_data[23, 1]))
gas_1300_O2 <- mean(c(freq_1300_data[3, 1], freq_1300_data[7, 1], freq_1300_data[11, 1], freq_1300_data[20, 1], freq_1300_data[24, 1]))
gas_1300_Ar <- mean(c(freq_1300_data[4, 1], freq_1300_data[8, 1], freq_1300_data[12, 1], freq_1300_data[16, 1], freq_1300_data[25, 1]))
gas_1300_Ar_O2 <- mean(c(freq_1300_data[5, 1], freq_1300_data[9, 1], freq_1300_data[13, 1], freq_1300_data[17, 1], freq_1300_data[21, 1]))
gas_1300_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_1300_He, gas_1300_Air, gas_1300_O2, gas_1300_Ar, gas_1300_Ar_O2))
names(gas_1300_meanS)[1]<- paste("Gas")
names(gas_1300_meanS)[2]<- paste("mean Sulfate (mM)")

temp_323_meanS
power_323_meanS
gas_323_meanS

temp_581_meanS
power_581_meanS
gas_581_meanS

temp_611_meanS
power_611_meanS
gas_611_meanS

temp_1300_meanS
power_1300_meanS
gas_1300_meanS


## Predicting the Unknown Data
