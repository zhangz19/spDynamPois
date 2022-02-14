
pois.scores <- function(pred.lambda, ho.counts){
  kk <- 100000 ## cut-off for summations 
  my.k <- (0:kk) - 1 ## to handle ranked probability score
  n <- length(ho.counts)
  
  p.pois.lambda <- pred.lambda
  p.pois.Px <- ppois(ho.counts, p.pois.lambda)
  p.pois.Px1 <- ppois(ho.counts-1, p.pois.lambda)
  p.pois.px <- dpois(ho.counts, p.pois.lambda)
  
  p.pois.logs <- -log(p.pois.px+.Machine$double.xmin)
  p.pois.norm <- 1:n
  for (i in 1:n) {p.pois.norm[i] <- sum(dpois(my.k,p.pois.lambda[i])^2)} 
  p.pois.qs <- -2*p.pois.px + p.pois.norm
  p.pois.sphs <- -p.pois.px / sqrt(p.pois.norm)
  p.pois.rps <- 1:n 
  for (i in 1:n) 
    {p.pois.rps[i] <- sum(ppois((-1):(ho.counts[i]-1),p.pois.lambda[i])^2) + sum((ppois(ho.counts[i]:kk,p.pois.lambda[i])-1)^2)}
  p.pois.dss <- (ho.counts-p.pois.lambda)^2/p.pois.lambda + log(p.pois.lambda)
  p.pois.rmse <- (ho.counts-p.pois.lambda)^2
  
  out <- c(mean(p.pois.logs),    ### logarithmic score
           mean(p.pois.qs),        ### quadratic score
           mean(p.pois.sphs),    ### spherical score  
           mean(p.pois.rps),      ### ranked probability score
           mean(p.pois.dss),      ### Dawid-Sebastiani score
           sqrt(mean(p.pois.rmse)))     ### root mean squared error score
  
  names(out) <- c("logarithmic", "quadratic", "spherical",
                  "ranked probability", "Dawid-Sebastiani", "root mean squared error")
  out
}


norm <- function(x){
  sqrt(sum(x^2))
}

crps <- function(y.hat,y){
  m <- ncol(y.hat)
  a <- 0
  b <- 0
  for(i in 1:m){
    a <- a+norm(y.hat[,i]-y)
    for(j in 1:m){
      b <- b+norm(y.hat[,i]-y.hat[,j])
    }
  }
  a/m-b/(2*(m^2))
}

rmse <- function(y.hat, y){
  sqrt(mean((y-y.hat)^2))
}

quant <- function(x){
  quantile(x, prob=c(0.025,0.975))
}

in.out <- function(q, y){
  if(y <= q[2] && y >= q[1]){1}else{0}
}

cover <- function(y.hat, y){
  q <- t(apply(y.hat, 2, quant))

  out <- rep(0, length(y))
  
  for(i in 1:length(y)){
    out[i] <- in.out(q[i,],y[i])
  }
  out
}

cover.summ <- function(x){
  c(length(x),
  round(100*sum(x)/length(x),0))
}

rng <- function(y.hat){
  q <- t(apply(y.hat, 1, quant))
  q[,2]-q[,1]
}
