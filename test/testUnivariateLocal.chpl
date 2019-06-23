use mklRandom;

// Common parameters
config const n = 2_000_000;
config const absThresh=0.2;
config const relThresh=0.02;
config const epsilon=1.0e-10;

var exitCode=0;

// Continuous distributions
runOne(new Uniform(-3.0,5.0), "Uniform", (1.0, 5.333,0.0,51.2),BRNG.SFMT19937, 10);
runOne(new Gaussian(1.0,1.0), "Gaussian", (1.0,1.0,0.0,3.0), BRNG.SFMT19937, 10);
runOne(new Exponential(0.0,1.0),"Exponential",(1.0,1.0,2.0,9.0),BRNG.SFMT19937,10);
runOne(new Laplace(3.0,2.0),"Laplace",(3.0,8.0,0.0,384.0),BRNG.SFMT19937,10);
runOne(new Weibull(2.0,2.0,1.0),"Weibull",(2.88623,0.214602,0.0627416,0.149449),BRNG.SFMT19937,10);
runOne(new Rayleigh(0.0,sqrt(2.0)),"Rayleigh",(1.25331,0.429204, 0.1774, 0.597797),BRNG.SFMT19937,10);
runOne(new Lognormal(0.0,0.5,0.0,1.0),"Lognormal",(1.13315, 0.364696, 0.385462, 1.18352),BRNG.SFMT19937,10);
runOne(new Gumbel(0.0,1.0),"Gumbel",(-0.577216, 1.64493, -2.40411, 14.6114),BRNG.SFMT19937,10);
runOne(new Gamma(4.0,0.0,1.0),"Gamma",(4.0, 4.0, 8.0, 72.0),BRNG.SFMT19937,10);
runOne(new Beta(0.3,0.7,0.0,1.0),"Beta",(0.3, 0.105, 0.028, 0.0249375),BRNG.SFMT19937,10);
runOne(new ChiSquare(3),"ChiSquare",(3.0,6.0,24.0,252.0),BRNG.SFMT19937,10);



// Cauchy
// The Cauchy distribution is different in that it doesn't have finite moments,
// so we need to test it another way.
runCauchy(BRNG.SFMT19937, 10);

// Discrete distributions
runOne(new UniformDiscrete(1,5),"UniformDiscrete",(2.5, 1.25, 0.0, 2.5625),BRNG.SFMT19937,10);
runOne(new Bernoulli(0.3),"Bernoulli",(0.3, 0.21, 0.084, 0.0777),BRNG.SFMT19937,10);
runOne(new Geometric(0.7),"Geometric",(0.428571, 0.612245, 1.13703, 3.98584),BRNG.SFMT19937,10);
runOne(new Binomial(7,0.3),"Binomial",(2.1, 1.47, 0.588, 6.1005),BRNG.SFMT19937,10);
runOne(new Hypergeometric(10,5,6),"Hypergeometric",(3.0, 0.666667, 0.0, 1.2381),BRNG.SFMT19937,10);
runOne(new Poisson(10.7),"Poisson(10.7)",(10.7,10.7,10.7,354.17),BRNG.SFMT19937,10);
runOne(new Poisson(1.2),"Poisson(1.2)",(1.2,1.2,1.2,5.52),BRNG.SFMT19937,10);
runOne(new Negbinomial(4.0,0.9),"Negbinomial",(0.444444, 0.493827, 0.603567, 1.59122),BRNG.SFMT19937,10);

exit(exitCode);


proc runOne(dist, name, expected, t : BRNG, iseed, istream=0) {
  var r1 = new MKLRandomStream(t, iseed, istream);
  var a : [0.. #n] dist.dtype;
  r1.fill(dist,a);
  stats(a, name, expected);
}

proc runCauchy(t : BRNG, iseed, istream=0) {
  var dist = new Cauchy(0.0,1.0);
  var r1 = new MKLRandomStream(t, iseed, istream);
  var a : [0.. #n] dist.dtype;
  r1.fill(dist,a);
  const val = (+ reduce sqrt(abs(a)))/n;
  writef("Cauchy          :  <sqrt(x)>=%10.3er, expected=%10.3er ",val,sqrt(2.0));
  writef(" "*11);
  const err = abs(val/sqrt(2.0)-1.0);
  if err > relThresh {
    writef("FAIL\n");
    exitCode += 1;
  } else {
    writef("PASS\n");
  }
}

proc stats(arr, name, expected) {
  const mean = (+ reduce arr)/arr.size:real;
  var dx = arr-mean;
  const m2 = (+ reduce dx**2)/arr.size;
  const m3 = (+ reduce dx**3)/arr.size;
  const m4 = (+ reduce dx**4)/arr.size;
  const tup = (mean, m2, m3, m4);
  var good=true;
  writef("%s%s : ",name," "*(15-name.length));
  for ii in 1..4 {
    good = good && test(tup(ii),expected(ii));
    writef("%10.3er ",tup(ii));
  }
  writef(" "*10);
  if good {
    writef("PASS\n");
  } else {
    writef("FAIL\n");
    exitCode += 1;
  }
}

proc test(x,y) {
  const err = abs(x-y);
  if (abs(y) < epsilon) then return err < absThresh;
  return err/abs(y) < relThresh;
}

