/* This module provides access to the MKL VSL random number generation libraries.

   Compilation
   -----------
   Include the appropriate flags for linking with MKL on your system. The MKL
   link line advisor is useful for this. This requires the 2019 Release 3 version
   of MKL or later.

   Types:
   ------
   For now, real types are assumed to be c_double==real(64) and integer types are
   c_int==int(32). In a lot of the library, we use the Chapel int, and the user
   is expected to ensure that these are appropriately bounded.

   Performance:
   ------------
   It is advantageous to generate a large number of random numbers at the same time. While
   we offer a method to generate single random numbers, that is not optimal from a performance
   standpoint.

   Parallelization strategies:
   ---------------------------
   NOTE : Given that the appropriate parallelization strategy is quite sensitive to the exact
   application, we do not implement a default parallelization strategy but leave that up to the
   user. We highlight some demonstrations of these in the test codes.

   NOTE : We recommend reading the Intel Notes on Vector Statistics for more details; we largely do not
   duplicate this information here.

   There are three parallelization strategies we support :

   1. Independent streams : One can initialize each stream differently. Also the WH and MT2203
      generators support multiple streams (273 and 6024 respectively) and we support initializing
      these with those streams.
   2. Skip Ahead : A large number of generators support skipping ahead. This is a convenient way
      to parallelize, if you know how many random numbers you need.
   3. Leap Frog : One can interleave the random number streams. This is useful when you know the
      number of streams required (and this is a relatively small number).

   NOTE : For the skipAhead methods, it is important to remember that this applies to the underlying
   stream and not the distribution. Generating a random variate from a distribution can sometimes
   require more than one basic random number.

   Distributed Arrays
   ------------------
   The fill method currently does not support distributed arrays. We leave it up to the user
   to fill these in. In particular, some care needs to be taken if reproducibility is to be guaranteed.
   We plan to include some examples of how to do this.

   Multivariate Distributions
   --------------------------
   MKL currently includes GaussianMV and Multinomial multivariate distributions. These are currently
   not implemented, although this is planned.

   In addition, it also includes PoissonV where the expectation lambda is a vector. This doesn't quite
   fit the usual distribution framework, and we plan on implementing this as a one-off array.

*/ 
prototype module mklRandom {
 

  extern {
    #include "mkl.h"
  }


  /* An error class for MKL library errors */
  class MKLError : Error {
    var status : c_int;

    proc init(s : c_int) {
      status=s;
    }

    override proc message() {
      return "MKL VSL Error : Status=%i".format(status);
    }
  }

  /* An error class for more generic errors */
  class RNGError : Error {
    var err : string;

    proc init(err1 : string) {
      err = err1;
    }

    override proc message() {
      return "RNG Error : %s".format(err);
    }
  }

  /* Define the basic random number engines.

     We currently don't implement all, especially
     since we require skipAhead.
  */
  enum BRNG {
             MRG32K3A,
             WH,
             MT19937,
             MT2203,
             SFMT19937
  };

  proc BRNG2Int(t : BRNG) {
    var ret : c_int;
    select t {
        when BRNG.MRG32K3A do ret=VSL_BRNG_MRG32K3A;
        when BRNG.WH do ret=VSL_BRNG_WH;
        when BRNG.MT19937 do ret=VSL_BRNG_MT19937;
        when BRNG.MT2203 do ret=VSL_BRNG_MT2203;
        when BRNG.SFMT19937 do ret=VSL_BRNG_SFMT19937;
      }
    return ret;
  }

  /* A holder for an MKL stream.

     This lets you directly work with the stream object, which
     can be useful if what you want to do doesn't quite fit into
     the workflows defined by the MKLRandomStream class.

     In particular, this is useful if one is using a leapfrog scheme.

     Since these are nothing but wrappers around a C pointer, these are
     by necessity local objects. Also note the record semantics, these aren't
     meant to be passed around.


   */
  record MKLRandomStream {
    var stream: VSLStreamStatePtr;

    /* Initialize with the MRG32K3A generator */
    proc init(iseed : integral) {
      init(BRNG.MRG32K3A, iseed);
    }

    /* Initialize with the appropriate type, with seed iseed.

       If the generator supports multiple streams, you can optionally initialize the stream.
       Streams run from 0.. #NStreamMax
    */
    proc init(t : BRNG, iseed : integral, istream : integral=0) {
      this.complete();

      if (istream > 0) {
        select t {
            when BRNG.WH do if (istream > 272) then halt("Error! WH only supports 272 streams.");
            when BRNG.MT2203 do if (istream > 6023) then halt("Error! MT2203 only supports 6023 streams.");
            otherwise do halt("Multiple streams not supported!");
          }
      }

      var status = vslNewStream(c_ptrTo(stream), BRNG2Int(t)+istream:c_int, iseed : c_uint);
      if (status != VSL_STATUS_OK) then halt("Error making MKLStreamPointer");
    }

    /* Save stream state. Returns an array with the stream state saved.*/
    proc save() throws {
      var size = vslGetStreamSize(stream) : int;
      var state : [0.. #size] c_char;
      var status = vslSaveStreamM(stream, c_ptrTo(state));
      if (status != VSL_STATUS_OK) then throw new owned MKLError(status);
      return state;
    }

    /* Load stream state. Updates the random stream */
    proc load(state : [?D] c_char) throws {
      var size = vslGetStreamSize(stream) : int;
      if D.size != size then throw new owned RNGError("Incorrect state vector passed in");
      var status = vslLoadStreamM(c_ptrTo(stream), c_ptrTo(state) : c_string);
      if (status != VSL_STATUS_OK) then throw new owned MKLError(status);
    }

    /* Skip ahead */
    proc skipAhead(skip: integral) throws {
      var status : c_int;
      status = vslSkipAheadStream(stream, skip : c_long);
      if (status != VSL_STATUS_OK) then throw new owned MKLError(status);
    }

    /* Initialize for leapfrogging.

       Not too many generators support this (see the MKL documentation),
       but it could be useful for quasi-random sequences).
    */
    proc leapFrog(istream : integral, nstreams : integral) throws {
      var status : c_int;
      status = vslLeapfrogStream(stream, istream : c_int, nstreams : c_int);
      if (status != VSL_STATUS_OK) then throw new owned MKLError(status);
    }

    /* Fill an array with random numbers from distribution dist*/
    proc fill(const dist, arr : [?Dom] ?T) throws {
      compilerAssert(dist.dtype==T,"Mismatched types between dist and arr");
      var n = arr.size : int(64);
      var status = dist.fill(stream, arr.size : c_int, c_ptrTo(arr));
      if (status != VSL_STATUS_OK) then throw new owned MKLError(status);
    }

    /* Generate an array of blocksize random numbers from distribution dist */
    proc generate(const dist, blocksize : integral) throws {
      var r : [1..blocksize] dist.dtype;
      var status=dist.fill(stream, blocksize : c_int, c_ptrTo(r));
      if (status != VSL_STATUS_OK) then throw new owned MKLError(status);
      return r;
    }

    /* Generate a single random number from distribution dist.

       This is not a very efficient way of testing the code.
     */
    proc generate(const dist) throws {
      var r : dist.dtype;
      var status=dist.fill(stream, 1, c_ptrTo(r));
      if (status != VSL_STATUS_OK) then throw new owned MKLError(status);
      return r;
    }

    proc deinit() {
      vslDeleteStream(c_ptrTo(stream));
    }
  }


  /* The definitions of distributions are below.

     These all have a similar structure, designed to be plugged in to
     the codes above. In particular, they all define a dtype method and
     a fill method. The fill method isn't meant to be used by an end-user.
  */
    

  /* Gaussian distribution, with mean mu and
     variance sigma.

     This is currently hardcoded to return doubles, and
     to use the ICDF method.
  */
  record Gaussian {
    var mu, sigma : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n,
                           arr, mu : c_double, sigma : c_double);
    }
  }

  /* Uniform distribution between [a,b)

  */
  record Uniform {
    var a,b : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, n,
                          arr, a : c_double, b : c_double);
    }

  }

  /* Exponential with displacement a and scale factor beta
  */
  record Exponential {
    var a,beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, stream, n,
                          arr, a : c_double, beta : c_double);
    }

  }

  /* Laplace with displacement a and scale factor beta
   */
  record Laplace {
    var a,beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngLaplace(VSL_RNG_METHOD_LAPLACE_ICDF, stream, n,
                          arr, a : c_double, beta : c_double);
    }

  }

  /* Weibull with displacement a, shape alpha, scale factor beta
   */
  record Weibull {
    var alpha, a,beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngWeibull(VSL_RNG_METHOD_WEIBULL_ICDF_ACCURATE, stream, n,
                          arr, alpha : c_double,
                          a : c_double, beta : c_double);
    }

  }

  /* Cauchy with displacement a, and scale factor beta
   */
  record Cauchy {
    var a,beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngCauchy(VSL_RNG_METHOD_CAUCHY_ICDF, stream, n,
                         arr, 
                         a : c_double, beta : c_double);
    }

  }

  /* Rayleigh with displacement a, and scale factor beta.

     The usual definition of the Rayleigh distribution has 2\sigma^2 in
     the denominator of the exponential, which here is just \beta^2.
   */
  record Rayleigh {
    var a,beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngRayleigh(VSL_RNG_METHOD_RAYLEIGH_ICDF_ACCURATE, stream, n,
                           arr, 
                           a : c_double, beta : c_double);
    }

  }

  /* Lognormal with a=average of subject normal, sigma=deviation of subject
     normal, b=displacement and beta=scalefactor.

     "a" and sigma are the usual parameters of a lognormal distribution,
     with b=0 and beta=1.
  */
  record Lognormal {
    var a, sigma : real;
    var b, beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF_ACCURATE, stream, n,
                            arr, 
                            a : c_double, sigma : c_double,
                            b : c_double, beta : c_double);
    }

  }

  /* Gumbel with displacement a, and scale factor beta
   */
  record Gumbel {
    var a,beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngGumbel(VSL_RNG_METHOD_GUMBEL_ICDF, stream, n,
                         arr, 
                         a : c_double, beta : c_double);
    }

  }

  /* Gamma with displacement a, shape alpha, scale factor beta
   */
  record Gamma {
    var alpha, a,beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, stream, n,
                        arr, alpha : c_double,
                        a : c_double, beta : c_double);
    }

  }

  /* Beta with p,q shape parameters, a=displacement, and beta scalefactor.

     Note that the canonical Beta distribution has a=0, beta=1. This version
     shifts the nonzero range to [a,a+beta)
   */
  record Beta {
    var p, q : real;
    var a, beta : real;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngBeta(VSL_RNG_METHOD_BETA_CJA_ACCURATE, stream, n,
                       arr, p : c_double, q : c_double,
                       a : c_double, beta : c_double);
    }

  }

  /* ChiSquare with nu degrees of freedom (int(32))
   */
  record ChiSquare {
    var nu : int;

    proc dtype type {
      return real;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_double)) 
    {
      return vdRngChiSquare(VSL_RNG_METHOD_CHISQUARE_CHI2GAMMA, stream, n,
                            arr, nu : c_int);
    }

  }

  /* Uniform discrete distribution, with integer bounds [a,b)

     Bounds should be max int(32).
  */
  record UniformDiscrete {
    var a,b : int;

    proc dtype type {
      return c_int;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_int)) 
    {
      return viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n,
                          arr, a : c_int, b : c_int);
    }

  }

  /* Bernoulli with success probability p of a trial
  */
  record Bernoulli {
    var p : real;

    proc dtype type {
      return c_int;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_int)) 
    {
      return viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, n,
                            arr, p : c_double);
    }

  }

  /* Geometric with success probability p of a trial
  */
  record Geometric {
    var p : real;

    proc dtype type {
      return c_int;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_int)) 
    {
      return viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF, stream, n,
                            arr, p : c_double);
    }

  }

  /* Binomial with ntrial trials, and success probability p of a trial
  */
  record Binomial {
    var ntrial : int;
    var p : real;

    proc dtype type {
      return c_int;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_int)) 
    {
      return viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, stream, n,
                           arr, ntrial : c_int, p : c_double);
    }

  }

  /* Hypergeometric with lot size l, size of sampling without
     replacement s, and number of marked elements m.
  */
  record Hypergeometric {
    var l : int;
    var s : int;
    var m : int;

    proc dtype type {
      return c_int;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_int)) 
    {
      return viRngHypergeometric(VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE, stream, n,
                                 arr,
                                 l : c_int, s : c_int, m : c_int);
    }

  }

  /* Poisson distribution, with scale factor lambda
  */
  record Poisson {
    var lam : real;

    proc dtype type {
      return c_int;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_int)) 
    {
      return viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM, stream, n,
                           arr, lam: c_double);
    }

  }

  /* Negative Binomial, with a and p distribution parameters.
  */
  record Negbinomial {
    var a : real;
    var p : real;

    proc dtype type {
      return c_int;
    }

    proc fill(stream : VSLStreamStatePtr, n : c_int, arr : c_ptr(c_int)) 
    {
      return viRngNegbinomial(VSL_RNG_METHOD_NEGBINOMIAL_NBAR, stream, n,
                              arr,
                              a : c_double, p : c_double);
    }

  }

}