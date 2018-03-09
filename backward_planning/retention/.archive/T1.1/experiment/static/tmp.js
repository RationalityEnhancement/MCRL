var webppl = require("/usr/local/lib/node_modules/webppl/src/main.js");
var __runner__ = util.trampolineRunners.cli();
function topK(s, x) {
  console.log(x);
};
var main = (function (_globalCurrentAddress) {
    return function (p) {
        return function (runTrampoline) {
            return function (s, k, a) {
                runTrampoline(function () {
                    return p(s, k, a);
                });
            };
        };
    }(function (globalStore, _k0, _address0) {
        var _currentAddress = _address0;
        _addr.save(_globalCurrentAddress, _address0);
        var _dummy1 = console.log('globalStore', globalStore);
        return function () {
            return _k0(globalStore, globalStore.show('hello'));
        };
    });
});

webppl.runEvaled(main, __runner__, {}, {}, topK, '');