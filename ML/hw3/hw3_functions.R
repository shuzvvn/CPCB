
require(dplyr)
require(MBCbook)

options(digits=18)

linear_forward<-function(x,weight,bias){
  x<-cbind(x0=rep(1,nrow(x)),x)
  theta<-cbind(intercept=bias,weight)
  linear_sum<-x %*% t(theta)
  return(linear_sum)
}

linear_backward<-function(input){
  d.weight<-input
  return(d.weight)
}

sigmoid_forward<-function(x){
  out<-1/(1+exp(-x))
  return(out)
}

sigmoid_backward<-function(x){
  d.a<-x*(1-x)
  return(d.a)
}

softmax_xeloss_forward<-function(x,labels){
  denominator<-apply(x,1,function(x)(sum(exp(x-20))))
  yhat<-exp(x-20)
  yhat<-sweep(yhat,1,denominator,'/')
  yhat.true.value<-matrix(yhat[cbind(seq_along(labels+1), labels+1)],ncol = 1)
  cross_entropy<--log(yhat.true.value)
  soft_xeloss.list<-list(yhat=yhat,cross_entropy=cross_entropy)
  return(soft_xeloss.list)
}


softmax_xeloss_backward<-function(yhat,labels){
  yhat.true.value<-as.numeric(yhat[cbind(seq_along(labels+1), labels+1)])
  d.b<-yhat
  d.b[cbind(seq_along(labels+1), labels+1)]<-yhat.true.value-1
  return(d.b)
}

neural_net_result<-function(x,alpha1,beta1,alpha2,beta2,y){
  x=as.matrix(x)
  a<-linear_forward(x = x,weight = alpha1,bias = beta1)
  z<-sigmoid_forward(a)
  b<-linear_forward(x=z, weight = alpha2,bias = beta2 )
  yhat<-softmax_xeloss_forward(x = b, labels = y)$yhat
  # minus one since the label starts from 0
  prediction<-as.numeric(apply(yhat,1,which.max))-1
  cross_entropy<-softmax_xeloss_forward(x = b, labels = y)$cross_entropy
  average_cross_entropy<-colMeans(cross_entropy)
  accuracy=sum(prediction==y)/length(y)
  neural_net_result=list(prediction=prediction,average_cross_entropy=average_cross_entropy,accuracy=accuracy)
  return(neural_net_result)
}


stochastic_gradient_descent_result<-function(train,
                                             eta=0.01,
                                             epoches=15,
                                             width.of.hidden.layer=256,
                                             batch.size=1,
                                             initial.alpha1,
                                             initial.beta1,
                                             initial.alpha2,
                                             initial.beta2,
                                             test){
  n=nrow(train)
  
  alpha1<-initial.alpha1[1:width.of.hidden.layer,] %>% as.matrix()
  beta1<-initial.beta1[1:width.of.hidden.layer,] %>% as.matrix()
  alpha2<-initial.alpha2[,1:width.of.hidden.layer] %>% as.matrix()
  beta2<-initial.beta2 %>% as.matrix()
  
  epoch.result.table<-data.frame(matrix(NA,ncol = 4,nrow = epoches))
  colnames(epoch.result.table)<-c('epoch','training.loss','testing.loss','testing.accuracy')
  
  beta2.list<-list()
  
  for (epoch in 1:epoches){
    
    index<-seq(1,n)
    mini.batch.index.list<-split(index, ceiling(seq_along(index) / batch.size))
    
    for (mini.batch in 1:length(mini.batch.index.list)){
      i=mini.batch.index.list[[mini.batch]]
  
      x=as.matrix(train[i,-785])
      y=train[i,785]
      
      a<-linear_forward(x = x,weight = alpha1,bias = beta1)
      z<-sigmoid_forward(a)
      b<-linear_forward(x=z, weight = alpha2,bias = beta2 )
      yhat<-softmax_xeloss_forward(x = b, labels = y)$yhat
      
      d.b<-softmax_xeloss_backward(yhat = yhat , labels = y)
      d.beta2<-matrix(apply(d.b,2,sum),ncol = 1)
      d.alpha2<-t(d.b) %*% linear_backward(z)
      d.z<-d.b %*% linear_backward(alpha2)
      d.a<-sigmoid_backward(x=z)
      d.beta1<-matrix(apply(d.z*d.a,2,sum),ncol = 1)
      d.alpha1<-t(d.z*d.a) %*% x
      
      beta2<-beta2-eta*d.beta2/batch.size
      alpha2<-alpha2-eta*d.alpha2/batch.size
      beta1<-beta1-eta*d.beta1/batch.size
      alpha1<-alpha1-eta*d.alpha1/batch.size
    }
    # compute average training loss using the updated alpha and beta in this epoch
    
    # beta2.list[[epoch]]<-beta2
    
    training.neural_net_result<-neural_net_result(x = train[,-785],
                                                  alpha1 = alpha1,
                                                  beta1 = beta1,
                                                  alpha2 = alpha2,
                                                  beta2 = beta2,
                                                  y = train[,785])
    testing.neural_net_result<-neural_net_result(x = test[,-785],
                                                 alpha1 = alpha1,
                                                 beta1 = beta1,
                                                 alpha2 = alpha2,
                                                 beta2 = beta2,
                                                 y = test[,785])
    epoch.result.table[epoch,1]<-epoch
    epoch.result.table[epoch,2]<-training.neural_net_result$average_cross_entropy
    epoch.result.table[epoch,3]<-testing.neural_net_result$average_cross_entropy
    epoch.result.table[epoch,4]<-testing.neural_net_result$accuracy
    print(paste('finish epoch:',epoch))

  }
  
  stochastic_gradient_descent_result.list<-list(epoch.result.table=epoch.result.table,
                                                alpha1=alpha1,
                                                beta1=beta1,
                                                alpha2=alpha2,
                                                beta2=beta2,
                                                beta2.list=beta2.list)
  return(stochastic_gradient_descent_result.list)

}

plot_image<-function(vector){
  vector<-as.numeric(vector)
  im<-t(matrix(vector,nrow = 28))
  imshow(im,col=palette(gray(0:255/255)))
}

hyperparameter.comparisonn.plot<-function(epoches,...,hyperparameter.name,hyperparameters,loss.of.interest){
  df.comparison<-data.frame(matrix(NA,ncol=4,nrow=0))
  colnames(df.comparison)<-c("epoch","training.loss","testing.loss","testing.accuracy")
  for (i in 1:length(list(...))){
    df=list(...)[[i]]$epoch.result.table
    df.comparison<-rbind(df.comparison,df)
  }
  df.comparison[,hyperparameter.name]<-rep(hyperparameters,each=epoches)
  df.comparison[,hyperparameter.name]<-as.factor(df.comparison[,hyperparameter.name])
  ggplot(df.comparison,aes_string('epoch',loss.of.interest,color=hyperparameter.name))+
    geom_path()+
    scale_color_brewer(palette = "Set2")+
    ggtitle(paste0(loss.of.interest,'.comparison'))+
    theme_classic()+
    theme(axis.text.x = element_text(size = 6),
          axis.text.y = element_text(size = 6),
          axis.title = element_text(size=6),
          legend.title=element_text(size=6), 
          legend.text=element_text(size=6),
          plot.title = element_text(hjust = 0.5,size=6))
}






