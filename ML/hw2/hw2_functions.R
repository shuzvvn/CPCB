
require(dplyr)
require(ggplot2)

##############################################################################################
# 1. Data processing functions
# binary variable encoding
binary_variable_encoding<-function(input,binary_variables){
  # input is a matrix/data.frame with features in columns and samples in rows
  # binary_variables is a vector indicating which binary variables to convert
  # set yes to zero and no to 1
  input[,binary_variables]<-sapply(input[,binary_variables],as.numeric)
  input[,binary_variables]=input[,binary_variables]-1
  return(input)
}

categorical_variable_encoding<-function(input,
                                        categorical_variable='ShelveLoc'){
  # input is a matrix/data.frame with features in columns and samples in rows
  # categorical_variable are categorical variables that with more than 2 values
  one_hot_encoding<-model.matrix(as.formula(paste0('~-1+',categorical_variable)),input) %>% as.data.frame()
  input<-input[,-which(colnames(input)==categorical_variable)]
  input<-cbind(input,one_hot_encoding)
  return(input)
}

feature_standardization<-function(input_train,input_test,response.feature,binary_variables,categorical_variable){
  # input_train and input_test should have same features
  input_test<-input_test[,colnames(input_train)]
  numeric.features.criteria<-sapply(input_train, is.numeric)
  numeric.features<-names(numeric.features.criteria)[numeric.features.criteria==T]
  numeric.features<-numeric.features[!numeric.features %in% c(response.feature,binary_variables,categorical_variable)]
  
  train.means<-apply(input_train[,numeric.features],2,mean)
  train.sds<-apply(input_train[,numeric.features],2,sd)
  
  input_train[,numeric.features]<-sweep(input_train[,numeric.features],2,train.means,'-')
  input_train[,numeric.features]<-sweep(input_train[,numeric.features],2,train.sds,'/')
  
  input_test[,numeric.features]<-sweep(input_test[,numeric.features],2,train.means,'-')
  input_test[,numeric.features]<-sweep(input_test[,numeric.features],2,train.sds,'/')
  
  out.list<-list(train_feature_standardized=input_train,
                 test_feature_standardized=input_test)
  return(out.list)
}

data_processing<-function(input_train,
                          input_test,
                          binary_variables,
                          categorical_variable='ShelveLoc',
                          response.feature='Sales'){
  # binary encoding
  input_train<-binary_variable_encoding(input_train,binary_variables)
  input_test<-binary_variable_encoding(input_test,binary_variables)
  
  # feature standardization
  feature_standardized.list<-feature_standardization(input_train =input_train,
                                    input_test = input_test,
                                    response.feature = response.feature,
                                    binary_variables=binary_variables,
                                    categorical_variable=categorical_variable)
  
  input_train=feature_standardized.list[['train_feature_standardized']]
  input_test=feature_standardized.list[['test_feature_standardized']]
  
  #categorical variable encoding
  output_train<-categorical_variable_encoding(input_train,categorical_variable)
  output_test<-categorical_variable_encoding(input_test,categorical_variable)
  
  # pull together into a list
  out.list<-list(training=output_train,
                 testing=output_test)
  
  return(out.list)
}
  
#############################################################################################
# 2. stochastic gradient descent functions
# given a dataset and parameter matrix, return loss function for the parameter
stochastic_gradient_descent_loss_function<-function(x,y,theta,regularization='none',lambda=NULL){
  # x is a data.frame with all the X variables in columns and contains only one row
  # y is the true value 
  # regularization can take from c('none','ridge','lasso')
  # however, in this assignment, we will only report (Ypred−Yactual)^2 value
  # theta is the parameter that we want to estimate, note: it is a combination of intercept and slopes
  # in order to estimate the slope the intercept simultaneously, we need to create a new column with all 1 in the data.point
  # the adding step should already by done, meaning that in x, it has a pseduo column 'intercept' 
  if(regularization=='none'){
    loss_function<-(x%*%theta-y)^2 %>% as.numeric()
  }
  if(regularization=='ridge'){
    theta_only_w<-theta[-1,] %>% as.matrix()
    loss_function<-(x%*%theta-y)^2 %>% as.numeric() + lambda*sum(theta_only_w^2)
  }
  if(regularization=='lasso'){
    theta_only_w<-theta[-1,] %>% as.matrix()
    loss_function<-(x%*%theta-y)^2 %>% as.numeric() + lambda*sum(abs(theta_only_w))
  }
  return(loss_function)
}


# build a table to record all the training loss in stochastic gradient descent
stochastic_gradient_descent_result<-function(train,Y,eta,epoch,regularization='none',lambda=NULL){
  # train is the data.frame with features in columns
  # Y is the Y variable that we want to predict
  # eta is the learning rate
  # epoch is number of epoch that we will perform in stochastic gradient descent
  # before we start the iteration, duplicate the training data to length nrow(train)*epoch
  # then we will go over each datapoint to calculate the loss function
  train=do.call("rbind", replicate(epoch, train, simplify = FALSE))
  
  x<-train[,-which(colnames(train)==Y)]
  y<-train[,Y]
  # append a pseudo column with all 1, this column will be used to estimate the intercept
  x<-cbind(rep(1,nrow(x)), x)
  x<-as.matrix(x)
  colnames(x)[1]<-'intercept'
  
  # build a loss table to store the loss function as each step as we iterate through the whole dataset
  loss.table<-data.frame(matrix(NA,ncol = 2,nrow = nrow(train)))
  colnames(loss.table)<-c('step','loss_function')
  
  # set initial theta
  # note: the first theta is intercept and the remaining ones are the slopes
  theta<-matrix(0,ncol(x),1)
  
  if(regularization=='none'){
    for (i in 1:nrow(x)){
      loss.table[i,1]=i
      loss.table[i,2]=stochastic_gradient_descent_loss_function(x[i,],y[i],theta = theta,regularization='none')
      # update theta
      y_hat<-as.numeric(x[i,] %*% theta - y[i])
      theta <- theta-2*eta*(y_hat*(as.matrix(x[i,])))
    }
  }
  if(regularization=='ridge'){
    for (i in 1:nrow(x)){
      loss.table[i,1]=i
      loss.table[i,2]=stochastic_gradient_descent_loss_function(x[i,],y[i],theta = theta,regularization='none')
      y_hat<-as.numeric(x[i,] %*% theta - y[i])
      theta_only_w<-theta[-1,] %>% as.matrix()
      theta_only_w_pseduo<-rbind(0,theta_only_w)
      # update theta
      theta <- theta - eta*(2*y_hat*(as.matrix(x[i,]))+2*lambda*theta_only_w_pseduo)
    }
  }
  if(regularization=='lasso'){
    for (i in 1:nrow(x)){
      loss.table[i,1]=i
      loss.table[i,2]=stochastic_gradient_descent_loss_function(x[i,],y[i],theta = theta,regularization='none')
      y_hat<-as.numeric(x[i,] %*% theta - y[i])
      theta_only_w<-theta[-1,] %>% as.matrix()
      # according to the note, when w=0, consider the derivative of w as 0
      theta_only_w_abs_derivative<-ifelse(theta_only_w>0,1,ifelse(theta_only_w<0,-1,0))
      theta_only_w_abs_derivative_pseduo<-rbind(0,theta_only_w_abs_derivative)
      # update theta
      theta<-theta-eta*(2*y_hat*(as.matrix(x[i,]))+lambda*theta_only_w_abs_derivative_pseduo)
    }
  }
  stochastic_gradient_descent_result<-list(loss.table=loss.table,
                                           theta=theta)
  return(stochastic_gradient_descent_result)
}

# given a loss.table, plot the loss curve at each step
plot_loss_curve<-function(stochastic_gradient_descent_table,title=NULL){
  p<-ggplot(stochastic_gradient_descent_table,aes(step,loss_function))+
    geom_line()+
    ggtitle(title)+
    theme_classic()
  plot(p)
}

##############################################################################
# 3. Evaluation functions
test_loss<-function(test,theta,Y){
  # test should have the same column as the training dataset
  # Y is the Y variable that we want to predict
  # theta is the parameter from training data, note: it is a combination of intercept and slopes
  true.value<-test[,Y]
  test<-test[,-which(colnames(test)==Y)]
  test<-as.matrix(test)
  test<-cbind(rep(1,nrow(test)),test)
  colnames(test)[1]<-'intercept'
  test_loss<-1/nrow(test)*sum((test%*%theta-true.value)^2)
  return(test_loss)
}

  
