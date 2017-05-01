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
temp_10 <- mean(scaled_df[21:25, 1])
temp_20 <- mean(scaled_df[6:10, 1])
temp_35 <- mean(scaled_df[11:15, 1])
temp_50 <- mean(scaled_df[16:20, 1])
temp_1300_meanS <- data.frame(temp = c(5, 10, 20, 35, 50), Predicted = c(temp_5, temp_10, temp_20, temp_35, temp_50))
names(temp_1300_meanS)[1]<- paste("Temperature")
names(temp_1300_meanS)[2]<- paste("Predicted (mM)")

## Power
power_51 <- mean(c(scaled_df[1, 1], scaled_df[6, 1], scaled_df[11, 1], scaled_df[16, 1], scaled_df[21, 1]))
power_90 <- mean(c(scaled_df[2, 1], scaled_df[7, 1], scaled_df[12, 1], scaled_df[17, 1], scaled_df[22, 1]))
power_137 <- mean(c(scaled_df[3, 1], scaled_df[8, 1], scaled_df[13, 1], scaled_df[18, 1], scaled_df[23, 1]))
power_181 <- mean(c(scaled_df[4, 1], scaled_df[9, 1], scaled_df[14, 1], scaled_df[19, 1], scaled_df[24, 1]))
power_223 <- mean(c(scaled_df[5, 1], scaled_df[10, 1], scaled_df[15, 1], scaled_df[20, 1], scaled_df[25, 1]))
power_1300_meanS <- data.frame(c(51, 90, 137, 181, 223), c(power_51, power_90, power_137, power_181, power_223))
names(power_1300_meanS)[1]<- paste("Power (W)")
names(power_1300_meanS)[2]<- paste("Predicted (mM)")

## Gas
gas_He <- mean(c(scaled_df[1, 1], scaled_df[10, 1], scaled_df[14, 1], scaled_df[18, 1], scaled_df[22, 1]))
gas_Air <- mean(c(scaled_df[2, 1], scaled_df[6, 1], scaled_df[15, 1], scaled_df[19, 1], scaled_df[23, 1]))
gas_O2 <- mean(c(scaled_df[3, 1], scaled_df[7, 1], scaled_df[11, 1], scaled_df[20, 1], scaled_df[24, 1]))
gas_Ar <- mean(c(scaled_df[4, 1], scaled_df[8, 1], scaled_df[12, 1], scaled_df[16, 1], scaled_df[25, 1]))
gas_Ar_O2 <- mean(c(scaled_df[5, 1], scaled_df[9, 1], scaled_df[13, 1], scaled_df[17, 1], scaled_df[21, 1]))
gas_1300_meanS <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(gas_He, gas_Air, gas_O2, gas_Ar, gas_Ar_O2))
names(gas_1300_meanS)[1]<- paste("Gas")
names(gas_1300_meanS)[2]<- paste("mean Sulfate (mM)")

min_error
best_neuron
best_learning_rate
scaled_df
temp_1300_meanS

################################### Experimental Data #####################################################
temp_1300_exp <- data.frame(c(5, 10, 20, 35, 50), c(0.5824, 0.7750, 0.7796, 0.8830, 0.5914))
power_1300_exp <- data.frame(c(51, 90, 137, 181, 223), c(0.8090, 0.9164, 0.6120, 0.7052, 0.5688))
gas_1300_exp <- data.frame(c('He', 'Air', 'O2', 'Ar', 'Ar/O2'), c(0.7918, 0.7722, 0.9738, 0.5602, 0.5134))

temp_1300_both <- data.frame(temp = temp_1300_exp[, 1], predicted = temp_1300_meanS[, 2], experimental = temp_1300_exp[, 2])
power_1300_both <- data.frame(power = power_1300_exp[, 1], predicted = power_1300_meanS[, 2], experimental = power_1300_exp[, 2])
gas_1300_both <-data.frame(gas = gas_1300_exp[, 1], predicted = gas_1300_meanS[, 2], experimental = gas_1300_exp[, 2])
#####################################MULTIPLE GRAPH FUNCTION ###############################
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

################################### Graph Predictive vs Actual ###########################
## Width: 781
## Height: 500
##
## Temperature
names(temp_1300_both) <- c("Temperature", "Predicted", "Experimental")
temp_1300_both.melt <- melt(temp_1300_both, id.vars = 'Temperature')
temp_1300_graph <- (
  ggplot(temp_1300_both.melt, aes(x = Temperature, 
                                  y = value, 
                                  shape = variable, 
                                  color = variable))
  + geom_point() 
  + geom_xspline(spline_shape=-0.4,size= 0.7)
  + scale_x_continuous(name = "Temperature (C)", breaks = 0:6 * 10, limits = c(0, 55))
  + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:24 * 0.1, limits = c(0, 0.95))
  #+ scale_color_manual(values = c("Predicted" = 'red','Experimental' = 'blue')) 
  + scale_shape_manual(values = c('Predicted' = 17, 'Experimental' = 16))
  + labs(color = "", shape = "")
)

## Power
names(power_1300_both) <- c("Power", "Predicted", "Experimental")
power_1300_both.melt <- melt(power_1300_both, id.vars = 'Power')
power_1300_graph <- (
  ggplot(power_1300_both.melt, aes(x = Power, 
                                   y = value, 
                                   shape = variable, 
                                   color = variable))
  + geom_point(size = 2) 
  + geom_xspline(spline_shape=-0.4,size= 0.7)
  + scale_x_continuous(name = "Power (W)", breaks = 0:6 * 50, limits = c(20, 250))
  + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:10 * 0.1, limits = c(0, 1))
  #+ scale_color_manual(values = c("Predicted" = 'red','Experimental' = 'blue')) 
  + scale_shape_manual(values = c('Predicted' = 17, 'Experimental' = 16))
  + labs(color = "", shape = "")
)

## Gas
gas_1300_both.melt <- melt(gas_1300_both, id.vars = 'gas')
gas_1300_graph <- (
  ggplot(data=gas_1300_both.melt, aes(x=gas, y=value, fill=variable)) 
  + geom_bar(stat="identity", position=position_dodge())
  + scale_y_continuous(name = "mean sulfate (mM)", breaks = 0:7 * 0.2, limits = c(0, 1.1))
  + labs(fill = "")
  + coord_cartesian(ylim=c(0, 1.1))
)
