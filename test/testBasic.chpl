use mklRandom;

var exitCode=0;

config const nblock=10;
config const blocksize=1000;
config const iseed=10;

testSkipAhead(BRNG.MRG32K3A);
testSkipAhead(BRNG.WH);
testSkipAhead(BRNG.MT19937);
testSkipAhead(BRNG.SFMT19937);
testSkipAhead(BRNG.WH,12);

testSkipAhead2(BRNG.MRG32K3A);
testSkipAhead2(BRNG.WH);
testSkipAhead2(BRNG.MT19937);
testSkipAhead2(BRNG.SFMT19937);
testSkipAhead2(BRNG.WH,12);

testLeapFrog(BRNG.WH);
testLeapFrog(BRNG.WH,12);


testLoadSave(BRNG.MRG32K3A);
testLoadSave(BRNG.WH);
testLoadSave(BRNG.MT19937);
testLoadSave(BRNG.SFMT19937);
testLoadSave(BRNG.WH,12);
testLoadSave(BRNG.MT2203);
testLoadSave(BRNG.MT2203,1023);

exit(exitCode);


// A simple demonstration of skipAhead, with a
// random Stream per iterate.
proc testLoadSave(t : BRNG, istream=0) {
  var a1,a2 : [0.. #(nblock*blocksize)] real;
  var r1 = new MKLRandomStream(t, iseed, istream);
  var state = r1.save();
  var dist = new Uniform(0.0,1.0);
  r1.fill(dist, a1);
  r1.load(state);
  r1.fill(dist, a2);
  var diff = max reduce abs(a1-a2);
  writef("Test  load save with %10s : max diff = %10.5er   %s\n",t, diff, passfail(diff, 1.0e-14));
}


// A simple demonstration of skipAhead, with a
// random Stream per iterate.
proc testSkipAhead(t : BRNG, istream=0) {
  var a1,a2 : [0.. #nblock, 0.. #blocksize] real;
  var r1 = new MKLRandomStream(t, iseed, istream);
  var dist = new Uniform(0.0,1.0);
  r1.fill(dist, a1);
  coforall ii in 0.. #nblock {
    var r2 = new MKLRandomStream(t, iseed, istream); // This is per-iteration.
    r2.skipAhead(ii*blocksize);
    r2.fill(dist,a2[ii,..]);
  }
  var diff = max reduce abs(a1-a2);
  writef("Test skip ahead with %10s : max diff = %10.5er   %s\n",t, diff, passfail(diff, 1.0e-14));
}

// A simple demonstration of skipAhead, with a
// random Stream per task
proc testSkipAhead2(t : BRNG, istream=0) {
  var a1,a2 : [0.. #nblock, 0.. #blocksize] real;
  var r1 = new MKLRandomStream(t, iseed, istream);
  var dist = new Uniform(0.0,1.0);
  r1.fill(dist, a1);
  forall ii in 0.. #nblock with (var r2 = new MKLRandomStream(t, iseed, istream),
                                 var curpos = 0) {
    r2.skipAhead(ii*blocksize-curpos);
    r2.fill(dist,a2[ii,..]);
    curpos = (ii+1)*blocksize;
  }
  var diff = max reduce abs(a1-a2);
  writef("Test skip ahead with %10s : max diff = %10.5er   %s\n",t, diff, passfail(diff, 1.0e-14));
}

// A simple demonstration of leapFrog, with a
// random Stream per task. 
proc testLeapFrog(t : BRNG, istream=0) {
  var a1,a2 : [0.. #(nblock*blocksize)] real;
  var r1 = new MKLRandomStream(t, iseed, istream);
  var dist = new Uniform(0.0,1.0);
  r1.fill(dist, a1);

  coforall itask in 0.. #nblock {
    var r2 = new MKLRandomStream(t, iseed, istream);
    r2.leapFrog(itask, nblock);
    r2.fill(dist, a2[(itask*blocksize).. #(blocksize)]);
  }

  var diff = 0.0;
  forall ii in 0.. #(nblock*blocksize) with (max reduce diff) {
    const itask = ii%nblock;
    const ndx = ii/nblock;
    diff = abs(a1[ii] - a2[itask*blocksize+ndx]);
  }
  writef("Test  leap frog with %10s : max diff = %10.5er   %s\n",t, diff, passfail(diff, 1.0e-14));
}


proc passfail(diff, thresh) {
  if diff > thresh {
    exitCode += 1;
    return "FAIL";
  }
  return "PASS";
}


