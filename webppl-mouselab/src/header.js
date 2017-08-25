module.exports = function(env) {
  // Header code goes here

  var callAsync = function(s, k, a, jsFunc, arg) {
    jsFunc(arg, function(returnVal) {
      global.resumeTrampoline(function() {
        return k(s, returnVal);
      });
    });
  };

  return {
    // Adjust exports here
    callAsync: callAsync
  }
}
