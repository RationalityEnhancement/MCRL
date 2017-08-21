var webppl = require("/usr/local/lib/node_modules/webppl/src/main.js");
var webpplMouselab = require("/Users/fred/Projects/mcrl/webppl-mouselab");
webppl.requireHeader("/Users/fred/Projects/mcrl/webppl-mouselab/src/header.js");
var webpplTimeit = require("/Users/fred/.webppl/node_modules/webppl-timeit");
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
        var Categorical = function Categorical(globalStore, _k560, _address7, params) {
            var _currentAddress = _address7;
            _addr.save(_globalCurrentAddress, _address7);
            return function () {
                return _k560(globalStore, util.jsnew(dists.Categorical, params));
            };
        };
        var idF = function idF(globalStore, _k422, _address71, x) {
            var _currentAddress = _address71;
            _addr.save(_globalCurrentAddress, _address71);
            return function () {
                return _k422(globalStore, x);
            };
        };
        var expectation = function expectation(globalStore, _k413, _address77, dist, func) {
            var _currentAddress = _address77;
            _addr.save(_globalCurrentAddress, _address77);
            var _k416 = function (globalStore, f) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                var supp = dist.support();
                return function () {
                    return reduce(globalStore, _k413, _address77.concat('_69'), function (globalStore, _k414, _address78, s, acc) {
                        var _currentAddress = _address78;
                        _addr.save(_globalCurrentAddress, _address78);
                        return function () {
                            return f(globalStore, function (globalStore, _result415) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return _k414(globalStore, ad.scalar.add(acc, ad.scalar.mul(ad.scalar.exp(dist.score(s)), _result415)));
                                };
                            }, _address78.concat('_68'), s);
                        };
                    }, 0, supp);
                };
            };
            return function () {
                return func ? _k416(globalStore, func) : _k416(globalStore, idF);
            };
        };
        var map_helper = function map_helper(globalStore, _k397, _address91, i, j, f) {
            var _currentAddress = _address91;
            _addr.save(_globalCurrentAddress, _address91);
            var n = ad.scalar.add(ad.scalar.sub(j, i), 1);
            return function () {
                return ad.scalar.eq(n, 0) ? _k397(globalStore, []) : ad.scalar.eq(n, 1) ? f(globalStore, function (globalStore, _result398) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return _k397(globalStore, [_result398]);
                    };
                }, _address91.concat('_70'), i) : function (globalStore, n1) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return map_helper(globalStore, function (globalStore, _result399) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return map_helper(globalStore, function (globalStore, _result400) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return _k397(globalStore, _result399.concat(_result400));
                                    };
                                }, _address91.concat('_72'), ad.scalar.add(i, n1), j, f);
                            };
                        }, _address91.concat('_71'), i, ad.scalar.sub(ad.scalar.add(i, n1), 1), f);
                    };
                }(globalStore, ad.scalar.ceil(ad.scalar.div(n, 2)));
            };
        };
        var map = function map(globalStore, _k395, _address92, fn, l) {
            var _currentAddress = _address92;
            _addr.save(_globalCurrentAddress, _address92);
            return function () {
                return map_helper(globalStore, _k395, _address92.concat('_74'), 0, ad.scalar.sub(l.length, 1), function (globalStore, _k396, _address93, i) {
                    var _currentAddress = _address93;
                    _addr.save(_globalCurrentAddress, _address93);
                    return function () {
                        return fn(globalStore, _k396, _address93.concat('_73'), l[i]);
                    };
                });
            };
        };
        var map2 = function map2(globalStore, _k393, _address94, fn, l1, l2) {
            var _currentAddress = _address94;
            _addr.save(_globalCurrentAddress, _address94);
            return function () {
                return map_helper(globalStore, _k393, _address94.concat('_76'), 0, ad.scalar.sub(l1.length, 1), function (globalStore, _k394, _address95, i) {
                    var _currentAddress = _address95;
                    _addr.save(_globalCurrentAddress, _address95);
                    return function () {
                        return fn(globalStore, _k394, _address95.concat('_75'), l1[i], l2[i]);
                    };
                });
            };
        };
        var extend = function extend(globalStore, _k384, _address102) {
            var _currentAddress = _address102;
            _addr.save(_globalCurrentAddress, _address102);
            var _arguments2 = Array.prototype.slice.call(arguments, 3);
            return function () {
                return _k384(globalStore, _.assign.apply(_, [{}].concat(_arguments2)));
            };
        };
        var reduce = function reduce(globalStore, _k381, _address103, fn, init, ar) {
            var _currentAddress = _address103;
            _addr.save(_globalCurrentAddress, _address103);
            var n = ar.length;
            var helper = function helper(globalStore, _k382, _address104, i) {
                var _currentAddress = _address104;
                _addr.save(_globalCurrentAddress, _address104);
                return function () {
                    return ad.scalar.peq(i, n) ? _k382(globalStore, init) : helper(globalStore, function (globalStore, _result383) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return fn(globalStore, _k382, _address104.concat('_84'), ar[i], _result383);
                        };
                    }, _address104.concat('_83'), ad.scalar.add(i, 1));
                };
            };
            return function () {
                return helper(globalStore, _k381, _address103.concat('_85'), 0);
            };
        };
        var sum = function sum(globalStore, _k379, _address105, l) {
            var _currentAddress = _address105;
            _addr.save(_globalCurrentAddress, _address105);
            return function () {
                return reduce(globalStore, _k379, _address105.concat('_86'), function (globalStore, _k380, _address106, a, b) {
                    var _currentAddress = _address106;
                    _addr.save(_globalCurrentAddress, _address106);
                    return function () {
                        return _k380(globalStore, ad.scalar.add(a, b));
                    };
                }, 0, l);
            };
        };
        var zip = function zip(globalStore, _k361, _address119, xs, ys) {
            var _currentAddress = _address119;
            _addr.save(_globalCurrentAddress, _address119);
            return function () {
                return map2(globalStore, _k361, _address119.concat('_98'), function (globalStore, _k362, _address120, x, y) {
                    var _currentAddress = _address120;
                    _addr.save(_globalCurrentAddress, _address120);
                    return function () {
                        return _k362(globalStore, [
                            x,
                            y
                        ]);
                    };
                }, xs, ys);
            };
        };
        var filter = function filter(globalStore, _k356, _address121, fn, ar) {
            var _currentAddress = _address121;
            _addr.save(_globalCurrentAddress, _address121);
            var helper = function helper(globalStore, _k357, _address122, i, j) {
                var _currentAddress = _address122;
                _addr.save(_globalCurrentAddress, _address122);
                var n = ad.scalar.add(ad.scalar.sub(j, i), 1);
                return function () {
                    return ad.scalar.eq(n, 0) ? _k357(globalStore, []) : ad.scalar.eq(n, 1) ? fn(globalStore, function (globalStore, _result358) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return _result358 ? _k357(globalStore, [ar[i]]) : _k357(globalStore, []);
                        };
                    }, _address122.concat('_99'), ar[i]) : function (globalStore, n1) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return helper(globalStore, function (globalStore, _result359) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return helper(globalStore, function (globalStore, _result360) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        return function () {
                                            return _k357(globalStore, _result359.concat(_result360));
                                        };
                                    }, _address122.concat('_101'), ad.scalar.add(i, n1), j);
                                };
                            }, _address122.concat('_100'), i, ad.scalar.sub(ad.scalar.add(i, n1), 1));
                        };
                    }(globalStore, ad.scalar.ceil(ad.scalar.div(n, 2)));
                };
            };
            return function () {
                return helper(globalStore, _k356, _address121.concat('_102'), 0, ad.scalar.sub(ar.length, 1));
            };
        };
        var maxWith = function maxWith(globalStore, _k343, _address129, f, ar) {
            var _currentAddress = _address129;
            _addr.save(_globalCurrentAddress, _address129);
            var fn = function fn(globalStore, _k346, _address130, _ar, _best) {
                var _currentAddress = _address130;
                _addr.save(_globalCurrentAddress, _address130);
                return function () {
                    return ad.scalar.peq(_ar.length, 0) ? _k346(globalStore, _best) : ad.scalar.gt(_ar[0][1], _best[1]) ? fn(globalStore, _k346, _address130.concat('_112'), _ar.slice(1), _ar[0]) : fn(globalStore, _k346, _address130.concat('_113'), _ar.slice(1), _best);
                };
            };
            return function () {
                return map(globalStore, function (globalStore, _result345) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return zip(globalStore, function (globalStore, _result344) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return fn(globalStore, _k343, _address129.concat('_116'), _result344, [
                                    ad.scalar.neg(Infinity),
                                    ad.scalar.neg(Infinity)
                                ]);
                            };
                        }, _address129.concat('_115'), ar, _result345);
                    };
                }, _address129.concat('_114'), f, ar);
            };
        };
        var repeat = function repeat(globalStore, _k323, _address135, n, fn) {
            var _currentAddress = _address135;
            _addr.save(_globalCurrentAddress, _address135);
            var helper = function helper(globalStore, _k330, _address136, m) {
                var _currentAddress = _address136;
                _addr.save(_globalCurrentAddress, _address136);
                return function () {
                    return ad.scalar.peq(m, 0) ? _k330(globalStore, []) : ad.scalar.peq(m, 1) ? fn(globalStore, function (globalStore, _result331) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return _k330(globalStore, [_result331]);
                        };
                    }, _address136.concat('_127')) : function (globalStore, m1) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        var m2 = ad.scalar.sub(m, m1);
                        return function () {
                            return helper(globalStore, function (globalStore, _result332) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return helper(globalStore, function (globalStore, _result333) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        return function () {
                                            return _k330(globalStore, _result332.concat(_result333));
                                        };
                                    }, _address136.concat('_129'), m2);
                                };
                            }, _address136.concat('_128'), m1);
                        };
                    }(globalStore, ad.scalar.ceil(ad.scalar.div(m, 2)));
                };
            };
            var _k327 = function (globalStore, _dummy326) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                var _k325 = function (globalStore, _dummy324) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return helper(globalStore, _k323, _address135.concat('_132'), n);
                    };
                };
                return function () {
                    return _.isFunction(fn) ? _k325(globalStore, undefined) : error(globalStore, _k325, _address135.concat('_131'), 'Expected second argument to be a function.');
                };
            };
            var _k329 = function (globalStore, _result328) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                return function () {
                    return _result328 ? error(globalStore, _k327, _address135.concat('_130'), 'Expected first argument to be a non-negative integer.') : _k327(globalStore, undefined);
                };
            };
            return function () {
                return util.isInteger(n) ? _k329(globalStore, ad.scalar.lt(n, 0)) : _k329(globalStore, !util.isInteger(n));
            };
        };
        var error = function error(globalStore, _k293, _address147, msg) {
            var _currentAddress = _address147;
            _addr.save(_globalCurrentAddress, _address147);
            return function () {
                return _k293(globalStore, util.error(msg));
            };
        };
        var SampleGuide = function SampleGuide(globalStore, _k289, _address151, wpplFn, options) {
            var _currentAddress = _address151;
            _addr.save(_globalCurrentAddress, _address151);
            return function () {
                return ForwardSample(globalStore, _k289, _address151.concat('_152'), wpplFn, _.assign({ guide: !0 }, _.omit(options, 'guide')));
            };
        };
        var OptimizeThenSample = function OptimizeThenSample(globalStore, _k287, _address152, wpplFn, options) {
            var _currentAddress = _address152;
            _addr.save(_globalCurrentAddress, _address152);
            return function () {
                return Optimize(globalStore, function (globalStore, _dummy288) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var opts = _.pick(options, 'samples', 'onlyMAP', 'verbose');
                    return function () {
                        return SampleGuide(globalStore, _k287, _address152.concat('_154'), wpplFn, opts);
                    };
                }, _address152.concat('_153'), wpplFn, _.omit(options, 'samples', 'onlyMAP'));
            };
        };
        var DefaultInfer = function DefaultInfer(globalStore, _k277, _address153, wpplFn, options) {
            var _currentAddress = _address153;
            _addr.save(_globalCurrentAddress, _address153);
            var _dummy286 = util.mergeDefaults(options, {}, 'Infer');
            var maxEnumTreeSize = 200000;
            var minSampleRate = 250;
            var samples = 1000;
            return function () {
                return Enumerate(globalStore, function (globalStore, enumResult) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var _k285 = function (globalStore, _dummy284) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        var _dummy283 = console.log('Using "rejection"');
                        return function () {
                            return Rejection(globalStore, function (globalStore, rejResult) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return rejResult instanceof Error ? function (globalStore, _dummy282) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        return function () {
                                            return CheckSampleAfterFactor(globalStore, function (globalStore, hasSampleAfterFactor) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                var _k280 = function (globalStore, _dummy279) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    var _dummy278 = console.log('Using "MCMC"');
                                                    return function () {
                                                        return MCMC(globalStore, _k277, _address153.concat('_161'), wpplFn, { samples: samples });
                                                    };
                                                };
                                                return function () {
                                                    return hasSampleAfterFactor ? function (globalStore, _dummy281) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return SMC(globalStore, function (globalStore, smcResult) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return dists.isDist(smcResult) ? _k277(globalStore, smcResult) : smcResult instanceof Error ? _k280(globalStore, console.log(ad.scalar.add(smcResult.message, '..quit SMC'))) : error(globalStore, _k280, _address153.concat('_160'), 'Invalid return value from SMC');
                                                                };
                                                            }, _address153.concat('_159'), wpplFn, {
                                                                throwOnError: !1,
                                                                particles: samples
                                                            });
                                                        };
                                                    }(globalStore, console.log('Using "SMC" (interleaving samples and factors detected)')) : _k280(globalStore, undefined);
                                                };
                                            }, _address153.concat('_158'), wpplFn);
                                        };
                                    }(globalStore, console.log(ad.scalar.add(rejResult.message, '..quit rejection'))) : dists.isDist(rejResult) ? _k277(globalStore, rejResult) : error(globalStore, _k277, _address153.concat('_162'), 'Invalid return value from rejection');
                                };
                            }, _address153.concat('_157'), wpplFn, {
                                minSampleRate: minSampleRate,
                                throwOnError: !1,
                                samples: samples
                            });
                        };
                    };
                    return function () {
                        return dists.isDist(enumResult) ? _k277(globalStore, enumResult) : enumResult instanceof Error ? _k285(globalStore, console.log(ad.scalar.add(enumResult.message, '..quit enumerate'))) : error(globalStore, _k285, _address153.concat('_156'), 'Invalid return value from enumerate');
                    };
                }, _address153.concat('_155'), wpplFn, {
                    maxEnumTreeSize: maxEnumTreeSize,
                    maxRuntimeInMS: 5000,
                    throwOnError: !1,
                    strategy: 'depthFirst'
                });
            };
        };
        var Infer = function Infer(globalStore, _k270, _address154, options, maybeFn) {
            var _currentAddress = _address154;
            _addr.save(_globalCurrentAddress, _address154);
            var _k276 = function (globalStore, wpplFn) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                var _k275 = function (globalStore, _dummy274) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var methodMap = {
                        SMC: SMC,
                        MCMC: MCMC,
                        PMCMC: PMCMC,
                        asyncPF: AsyncPF,
                        rejection: Rejection,
                        enumerate: Enumerate,
                        incrementalMH: IncrementalMH,
                        forward: ForwardSample,
                        optimize: OptimizeThenSample,
                        defaultInfer: DefaultInfer
                    };
                    var _k273 = function (globalStore, methodName) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        var _k272 = function (globalStore, _dummy271) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            var method = methodMap[methodName];
                            return function () {
                                return method(globalStore, _k270, _address154.concat('_165'), wpplFn, _.omit(options, 'method', 'model'));
                            };
                        };
                        return function () {
                            return _.has(methodMap, methodName) ? _k272(globalStore, undefined) : function (globalStore, methodNames) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                var msg = ad.scalar.add(ad.scalar.add(ad.scalar.add(ad.scalar.add('Infer: \'', methodName), '\' is not a valid method. The following methods are available: '), methodNames.join(', ')), '.');
                                return function () {
                                    return error(globalStore, _k272, _address154.concat('_164'), msg);
                                };
                            }(globalStore, _.keys(methodMap));
                        };
                    };
                    return function () {
                        return options.method ? _k273(globalStore, options.method) : _k273(globalStore, 'defaultInfer');
                    };
                };
                return function () {
                    return _.isFunction(wpplFn) ? _k275(globalStore, undefined) : error(globalStore, _k275, _address154.concat('_163'), 'Infer: a model was not specified.');
                };
            };
            return function () {
                return util.isObject(options) ? maybeFn ? _k276(globalStore, maybeFn) : _k276(globalStore, options.model) : _k276(globalStore, options);
            };
        };
        var utils = webpplMouselab;
        var TERM_ACTION = '__TERM_ACTION__';
        var TERM_STATE = '__TERM_STATE__';
        var UNKNOWN = '__';
        var INITIAL_NODE = '';
        var vals = function vals(globalStore, _k153, _address188, mu, sigma) {
            var _currentAddress = _address188;
            _addr.save(_globalCurrentAddress, _address188);
            return function () {
                return map(globalStore, _k153, _address188.concat('_248'), function (globalStore, _k154, _address189, x) {
                    var _currentAddress = _address189;
                    _addr.save(_globalCurrentAddress, _address189);
                    return function () {
                        return _k154(globalStore, ad.scalar.add(mu, ad.scalar.mul(x, sigma)));
                    };
                }, [
                    ad.scalar.neg(2),
                    ad.scalar.neg(1),
                    1,
                    2
                ]);
            };
        };
        var probs = function probs(globalStore, _k152, _address190) {
            var _currentAddress = _address190;
            _addr.save(_globalCurrentAddress, _address190);
            return function () {
                return _k152(globalStore, [
                    0.15,
                    0.35,
                    0.35,
                    0.15
                ]);
            };
        };
        var _dummy151 = globalStore.cost = 0;
        return function () {
            return vals(globalStore, function (globalStore, _result149) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                return function () {
                    return probs(globalStore, function (globalStore, _result150) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return Categorical(globalStore, function (globalStore, _result148) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                var _dummy147 = globalStore.reward = _result148;
                                var env = utils.buildEnv([
                                    2,
                                    2
                                ]);
                                var _dummy146 = console.log('env', env);
                                var tree = env.tree;
                                var nodes = _.range(tree.length);
                                var children = function children(globalStore, _k145, _address191, node) {
                                    var _currentAddress = _address191;
                                    _addr.save(_globalCurrentAddress, _address191);
                                    return function () {
                                        return _k145(globalStore, tree[node]);
                                    };
                                };
                                var nodeReward = function nodeReward(globalStore, _k144, _address192, node) {
                                    var _currentAddress = _address192;
                                    _addr.save(_globalCurrentAddress, _address192);
                                    return function () {
                                        return ad.scalar.eq(node, INITIAL_NODE) ? _k144(globalStore, 0) : sample(globalStore, _k144, _address192.concat('_252'), globalStore.reward);
                                    };
                                };
                                var expectedNodeReward = function expectedNodeReward(globalStore, _k143, _address193, state, node) {
                                    var _currentAddress = _address193;
                                    _addr.save(_globalCurrentAddress, _address193);
                                    return function () {
                                        return ad.scalar.eq(node, INITIAL_NODE) ? _k143(globalStore, 0) : ad.scalar.eq(state[node], UNKNOWN) ? expectation(globalStore, _k143, _address193.concat('_253'), globalStore.reward) : _k143(globalStore, state[node]);
                                    };
                                };
                                var nodeQuality = dp.cache(function (globalStore, _k136, _address194, state, node) {
                                    var _currentAddress = _address194;
                                    _addr.save(_globalCurrentAddress, _address194);
                                    return function () {
                                        return children(globalStore, function (globalStore, _result138) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            var _k139 = function (globalStore, best_child_val) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return expectedNodeReward(globalStore, function (globalStore, _result137) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return _k136(globalStore, ad.scalar.add(_result137, best_child_val));
                                                        };
                                                    }, _address194.concat('_258'), state, node);
                                                };
                                            };
                                            return function () {
                                                return ad.scalar.eq(_result138.length, 0) ? _k139(globalStore, 0) : children(globalStore, function (globalStore, _result142) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    return function () {
                                                        return map(globalStore, function (globalStore, _result140) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return _k139(globalStore, _.max(_result140));
                                                            };
                                                        }, _address194.concat('_257'), function (globalStore, _k141, _address195, child) {
                                                            var _currentAddress = _address195;
                                                            _addr.save(_globalCurrentAddress, _address195);
                                                            return function () {
                                                                return nodeQuality(globalStore, _k141, _address195.concat('_255'), state, child);
                                                            };
                                                        }, _result142);
                                                    };
                                                }, _address194.concat('_256'), node);
                                            };
                                        }, _address194.concat('_254'), node);
                                    };
                                });
                                var _dummy135 = globalStore.energySpent = 1;
                                var _dummy134 = globalStore.rewardAccrued = 0;
                                var termReward = function termReward(globalStore, _k133, _address196, state) {
                                    var _currentAddress = _address196;
                                    _addr.save(_globalCurrentAddress, _address196);
                                    return function () {
                                        return nodeQuality(globalStore, function (globalStore, expectedReward) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            return function () {
                                                return _k133(globalStore, expectedReward);
                                            };
                                        }, _address196.concat('_259'), state, INITIAL_NODE);
                                    };
                                };
                                var transition = function transition(globalStore, _k123, _address197, state, action) {
                                    var _currentAddress = _address197;
                                    _addr.save(_globalCurrentAddress, _address197);
                                    var _k132 = function (globalStore, _dummy131) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        var _k130 = function (globalStore, _dummy129) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            var _k126 = function (globalStore, _dummy125) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return nodeReward(globalStore, function (globalStore, _result124) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return _k123(globalStore, utils.updateList(state, action, _result124));
                                                        };
                                                    }, _address197.concat('_263'), action);
                                                };
                                            };
                                            return function () {
                                                return ad.scalar.neq(state[action], UNKNOWN) ? actions(globalStore, function (globalStore, _result128) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    var _dummy127 = console.log(_result128);
                                                    return function () {
                                                        return error(globalStore, _k126, _address197.concat('_262'), ad.scalar.add(ad.scalar.add(ad.scalar.add('observing state twice\n', JSON.stringify(state)), ' '), action));
                                                    };
                                                }, _address197.concat('_261'), state) : _k126(globalStore, undefined);
                                            };
                                        };
                                        return function () {
                                            return ad.scalar.eq(action, TERM_ACTION) ? _k123(globalStore, TERM_STATE) : _k130(globalStore, undefined);
                                        };
                                    };
                                    return function () {
                                        return ad.scalar.eq(state, TERM_STATE) ? error(globalStore, _k132, _address197.concat('_260'), ad.scalar.add('transition from term ', action)) : _k132(globalStore, undefined);
                                    };
                                };
                                var reward = function reward(globalStore, _k122, _address198, state, action) {
                                    var _currentAddress = _address198;
                                    _addr.save(_globalCurrentAddress, _address198);
                                    return function () {
                                        return ad.scalar.eq(action, TERM_ACTION) ? termReward(globalStore, _k122, _address198.concat('_264'), state) : _k122(globalStore, globalStore.cost);
                                    };
                                };
                                var unobservedNodes = function unobservedNodes(globalStore, _k120, _address199, state) {
                                    var _currentAddress = _address199;
                                    _addr.save(_globalCurrentAddress, _address199);
                                    return function () {
                                        return filter(globalStore, _k120, _address199.concat('_265'), function (globalStore, _k121, _address200, node) {
                                            var _currentAddress = _address200;
                                            _addr.save(_globalCurrentAddress, _address200);
                                            return function () {
                                                return _k121(globalStore, ad.scalar.eq(state[node], UNKNOWN));
                                            };
                                        }, nodes);
                                    };
                                };
                                var actions = function actions(globalStore, _k115, _address203, state) {
                                    var _currentAddress = _address203;
                                    _addr.save(_globalCurrentAddress, _address203);
                                    return function () {
                                        return unobservedNodes(globalStore, function (globalStore, _result116) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            return function () {
                                                return _k115(globalStore, _result116.concat([TERM_ACTION]));
                                            };
                                        }, _address203.concat('_267'), state);
                                    };
                                };
                                var enumPolicy = function enumPolicy(globalStore, _k86, _address210, opts) {
                                    var _currentAddress = _address210;
                                    _addr.save(_globalCurrentAddress, _address210);
                                    return function () {
                                        return extend(globalStore, function (globalStore, params) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            var myActions = params.myActions;
                                            var actionAndValue = dp.cache(function (globalStore, _k93, _address211, state) {
                                                var _currentAddress = _address211;
                                                _addr.save(_globalCurrentAddress, _address211);
                                                var Q = function Q(globalStore, _k101, _address212, action) {
                                                    var _currentAddress = _address212;
                                                    _addr.save(_globalCurrentAddress, _address212);
                                                    return function () {
                                                        return Infer(globalStore, function (globalStore, _result102) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return expectation(globalStore, _k101, _address212.concat('_280'), _result102);
                                                            };
                                                        }, _address212.concat('_279'), {
                                                            model: function (globalStore, _k103, _address213) {
                                                                var _currentAddress = _address213;
                                                                _addr.save(_globalCurrentAddress, _address213);
                                                                return function () {
                                                                    return transition(globalStore, function (globalStore, newState) {
                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                        return function () {
                                                                            return reward(globalStore, function (globalStore, _result104) {
                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                return function () {
                                                                                    return V(globalStore, function (globalStore, _result105) {
                                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                        return function () {
                                                                                            return _k103(globalStore, ad.scalar.add(_result104, _result105));
                                                                                        };
                                                                                    }, _address213.concat('_278'), newState);
                                                                                };
                                                                            }, _address213.concat('_277'), state, action);
                                                                        };
                                                                    }, _address213.concat('_276'), state, action);
                                                                };
                                                            },
                                                            method: 'enumerate'
                                                        });
                                                    };
                                                };
                                                var _k99 = function (globalStore, _dummy98) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    return function () {
                                                        return myActions(globalStore, function (globalStore, _result97) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return maxWith(globalStore, function (globalStore, result) {
                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                    var _k95 = function (globalStore, _dummy94) {
                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                        return function () {
                                                                            return _k93(globalStore, result);
                                                                        };
                                                                    };
                                                                    return function () {
                                                                        return ad.scalar.eq(result[0], ad.scalar.neg(Infinity)) ? myActions(globalStore, function (globalStore, _result96) {
                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                            return function () {
                                                                                return error(globalStore, _k95, _address211.concat('_286'), ad.scalar.add('problem!\n', _result96));
                                                                            };
                                                                        }, _address211.concat('_285'), state) : _k95(globalStore, undefined);
                                                                    };
                                                                }, _address211.concat('_284'), Q, _result97);
                                                            };
                                                        }, _address211.concat('_283'), state);
                                                    };
                                                };
                                                return function () {
                                                    return myActions(globalStore, function (globalStore, _result100) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return ad.scalar.eq(_result100.length, 0) ? error(globalStore, _k99, _address211.concat('_282'), 'no actions') : _k99(globalStore, undefined);
                                                        };
                                                    }, _address211.concat('_281'), state);
                                                };
                                            });
                                            var V = utils.cache(function (globalStore, _k91, _address214, state) {
                                                var _currentAddress = _address214;
                                                _addr.save(_globalCurrentAddress, _address214);
                                                return function () {
                                                    return ad.scalar.eq(state, TERM_STATE) ? _k91(globalStore, 0) : actionAndValue(globalStore, function (globalStore, _result92) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return _k91(globalStore, _result92[1]);
                                                        };
                                                    }, _address214.concat('_287'), state);
                                                };
                                            }, env);
                                            var policy = function policy(globalStore, _k89, _address215, state) {
                                                var _currentAddress = _address215;
                                                _addr.save(_globalCurrentAddress, _address215);
                                                return function () {
                                                    return actionAndValue(globalStore, function (globalStore, _result90) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        var a = _result90[0];
                                                        return function () {
                                                            return _k89(globalStore, a);
                                                        };
                                                    }, _address215.concat('_288'), state);
                                                };
                                            };
                                            return function () {
                                                return _k86(globalStore, policy);
                                            };
                                        }, _address210.concat('_275'), {
                                            maxExecutions: Infinity,
                                            alpha: 1000,
                                            myActions: actions
                                        }, opts);
                                    };
                                };
                                var simulate = function simulate(globalStore, _k73, _address222, policy) {
                                    var _currentAddress = _address222;
                                    _addr.save(_globalCurrentAddress, _address222);
                                    var rec = function rec(globalStore, _k74, _address223, acc) {
                                        var _currentAddress = _address223;
                                        _addr.save(_globalCurrentAddress, _address223);
                                        var state = _.last(acc.states);
                                        return function () {
                                            return ad.scalar.eq(state, TERM_STATE) ? _k74(globalStore, acc) : policy(globalStore, function (globalStore, action) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return transition(globalStore, function (globalStore, newState) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return reward(globalStore, function (globalStore, r) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return rec(globalStore, _k74, _address223.concat('_303'), {
                                                                        states: acc.states.concat([newState]),
                                                                        rewards: acc.rewards.concat([r]),
                                                                        actions: acc.actions.concat([action])
                                                                    });
                                                                };
                                                            }, _address223.concat('_302'), state, action);
                                                        };
                                                    }, _address223.concat('_301'), state, action);
                                                };
                                            }, _address223.concat('_300'), state);
                                        };
                                    };
                                    return function () {
                                        return rec(globalStore, _k73, _address222.concat('_304'), {
                                            states: [env.initialState],
                                            rewards: [],
                                            actions: []
                                        });
                                    };
                                };
                                return function () {
                                    return Categorical(globalStore, function (globalStore, REWARD) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        var value = function value(globalStore, _k72, _address224, tree) {
                                            var _currentAddress = _address224;
                                            _addr.save(_globalCurrentAddress, _address224);
                                            return function () {
                                                return _k72(globalStore, tree[0]);
                                            };
                                        };
                                        var children = function children(globalStore, _k71, _address225, tree) {
                                            var _currentAddress = _address225;
                                            _addr.save(_globalCurrentAddress, _address225);
                                            return function () {
                                                return _k71(globalStore, tree[1]);
                                            };
                                        };
                                        var OBSERVE = '__OBSERVE__';
                                        var HIDDEN = '__HIDDEN__';
                                        return function () {
                                            return repeat(globalStore, function (globalStore, _result69) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                var initialState = [0].concat(_result69);
                                                var subjectiveReward = function subjectiveReward(globalStore, _k68, _address227, tree) {
                                                    var _currentAddress = _address227;
                                                    _addr.save(_globalCurrentAddress, _address227);
                                                    return function () {
                                                        return value(globalStore, function (globalStore, v) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return ad.scalar.eq(v, HIDDEN) ? expectation(globalStore, _k68, _address227.concat('_308'), REWARD) : ad.scalar.eq(v, OBSERVE) ? sample(globalStore, _k68, _address227.concat('_309'), REWARD) : _k68(globalStore, v);
                                                            };
                                                        }, _address227.concat('_307'), tree);
                                                    };
                                                };
                                                var observationValue = dp.cache(function (globalStore, _k58, _address228, tree, params) {
                                                    var _currentAddress = _address228;
                                                    _addr.save(_globalCurrentAddress, _address228);
                                                    return function () {
                                                        return extend(globalStore, function (globalStore, params) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return extend(globalStore, function (globalStore, _result59) {
                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                    return function () {
                                                                        return Infer(globalStore, _k58, _address228.concat('_318'), _result59);
                                                                    };
                                                                }, _address228.concat('_317'), params, {
                                                                    model: function (globalStore, _k60, _address229) {
                                                                        var _currentAddress = _address229;
                                                                        _addr.save(_globalCurrentAddress, _address229);
                                                                        return function () {
                                                                            return children(globalStore, function (globalStore, _result62) {
                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                var _k63 = function (globalStore, bestChildVal) {
                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                    return function () {
                                                                                        return subjectiveReward(globalStore, function (globalStore, _result61) {
                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                            return function () {
                                                                                                return _k60(globalStore, ad.scalar.add(_result61, bestChildVal));
                                                                                            };
                                                                                        }, _address229.concat('_316'), tree);
                                                                                    };
                                                                                };
                                                                                return function () {
                                                                                    return ad.scalar.eq(_result62.length, 0) ? _k63(globalStore, 0) : children(globalStore, function (globalStore, _result67) {
                                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                        return function () {
                                                                                            return map(globalStore, function (globalStore, _result64) {
                                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                return function () {
                                                                                                    return _k63(globalStore, _.max(_result64));
                                                                                                };
                                                                                            }, _address229.concat('_315'), function (globalStore, _k65, _address230, child) {
                                                                                                var _currentAddress = _address230;
                                                                                                _addr.save(_globalCurrentAddress, _address230);
                                                                                                return function () {
                                                                                                    return observationValue(globalStore, function (globalStore, _result66) {
                                                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                        return function () {
                                                                                                            return sample(globalStore, _k65, _address230.concat('_313'), _result66);
                                                                                                        };
                                                                                                    }, _address230.concat('_312'), child, params);
                                                                                                };
                                                                                            }, _result67);
                                                                                        };
                                                                                    }, _address229.concat('_314'), tree);
                                                                                };
                                                                            }, _address229.concat('_311'), tree);
                                                                        };
                                                                    }
                                                                });
                                                            };
                                                        }, _address228.concat('_310'), { method: 'enumerate' }, params);
                                                    };
                                                });
                                                var expectedObservationValue = function expectedObservationValue(globalStore, _k54, _address231, tree, params) {
                                                    var _currentAddress = _address231;
                                                    _addr.save(_globalCurrentAddress, _address231);
                                                    var _dummy57 = console.log('EOV', JSON.stringify(tree));
                                                    var StartTime = Date.now();
                                                    return function () {
                                                        return observationValue(globalStore, function (globalStore, _result56) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return expectation(globalStore, function (globalStore, result) {
                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                    var _dummy55 = console.log('observationValue:', ad.scalar.div(ad.scalar.round(ad.scalar.mul(result, 100)), 100), '  time:', ad.scalar.sub(Date.now(), StartTime));
                                                                    return function () {
                                                                        return _k54(globalStore, result);
                                                                    };
                                                                }, _address231.concat('_320'), _result56);
                                                            };
                                                        }, _address231.concat('_319'), tree, params);
                                                    };
                                                };
                                                var stateTree = function stateTree(globalStore, _k53, _address232, state) {
                                                    var _currentAddress = _address232;
                                                    _addr.save(_globalCurrentAddress, _address232);
                                                    var s = state;
                                                    return function () {
                                                        return _k53(globalStore, [
                                                            s[0],
                                                            [
                                                                [
                                                                    s[1],
                                                                    [[
                                                                            s[5],
                                                                            [
                                                                                [
                                                                                    s[9],
                                                                                    []
                                                                                ],
                                                                                [
                                                                                    s[10],
                                                                                    []
                                                                                ]
                                                                            ]
                                                                        ]]
                                                                ],
                                                                [
                                                                    s[2],
                                                                    [[
                                                                            s[6],
                                                                            [
                                                                                [
                                                                                    s[11],
                                                                                    []
                                                                                ],
                                                                                [
                                                                                    s[12],
                                                                                    []
                                                                                ]
                                                                            ]
                                                                        ]]
                                                                ],
                                                                [
                                                                    s[3],
                                                                    [[
                                                                            s[7],
                                                                            [
                                                                                [
                                                                                    s[13],
                                                                                    []
                                                                                ],
                                                                                [
                                                                                    s[14],
                                                                                    []
                                                                                ]
                                                                            ]
                                                                        ]]
                                                                ],
                                                                [
                                                                    s[4],
                                                                    [[
                                                                            s[8],
                                                                            [
                                                                                [
                                                                                    s[15],
                                                                                    []
                                                                                ],
                                                                                [
                                                                                    s[16],
                                                                                    []
                                                                                ]
                                                                            ]
                                                                        ]]
                                                                ]
                                                            ]
                                                        ]);
                                                    };
                                                };
                                                return function () {
                                                    return cache(globalStore, function (globalStore, termValue) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        var timeit = function timeit(globalStore, _k32, _address241, thunk) {
                                                            var _currentAddress = _address241;
                                                            _addr.save(_globalCurrentAddress, _address241);
                                                            var t0 = webpplTimeit.now();
                                                            return function () {
                                                                return thunk(globalStore, function (globalStore, value) {
                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                    var t1 = webpplTimeit.now();
                                                                    return function () {
                                                                        return _k32(globalStore, {
                                                                            value: value,
                                                                            runtimeInMilliseconds: ad.scalar.sub(t1, t0)
                                                                        });
                                                                    };
                                                                }, _address241.concat('_335'));
                                                            };
                                                        };
                                                        var L = webpplMouselab;
                                                        var round = function round(globalStore, _k30, _address243, x, p) {
                                                            var _currentAddress = _address243;
                                                            _addr.save(_globalCurrentAddress, _address243);
                                                            return function () {
                                                                return _k30(globalStore, ad.scalar.div(ad.scalar.round(ad.scalar.mul(x, ad.scalar.pow(10, p))), ad.scalar.pow(10, p)));
                                                            };
                                                        };
                                                        var vals = function vals(globalStore, _k28, _address244, mu, sigma) {
                                                            var _currentAddress = _address244;
                                                            _addr.save(_globalCurrentAddress, _address244);
                                                            return function () {
                                                                return map(globalStore, _k28, _address244.concat('_336'), function (globalStore, _k29, _address245, x) {
                                                                    var _currentAddress = _address245;
                                                                    _addr.save(_globalCurrentAddress, _address245);
                                                                    return function () {
                                                                        return _k29(globalStore, ad.scalar.add(mu, ad.scalar.mul(x, sigma)));
                                                                    };
                                                                }, [
                                                                    ad.scalar.neg(2),
                                                                    ad.scalar.neg(1),
                                                                    1,
                                                                    2
                                                                ]);
                                                            };
                                                        };
                                                        var probs = function probs(globalStore, _k27, _address246) {
                                                            var _currentAddress = _address246;
                                                            _addr.save(_globalCurrentAddress, _address246);
                                                            return function () {
                                                                return _k27(globalStore, [
                                                                    0.15,
                                                                    0.35,
                                                                    0.35,
                                                                    0.15
                                                                ]);
                                                            };
                                                        };
                                                        var _dummy26 = globalStore.cost = 0;
                                                        return function () {
                                                            return vals(globalStore, function (globalStore, _result24) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return probs(globalStore, function (globalStore, _result25) {
                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                        return function () {
                                                                            return Categorical(globalStore, function (globalStore, _result23) {
                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                var _dummy22 = globalStore.reward = _result23;
                                                                                var run = function run(globalStore, _k9, _address247, name, policy) {
                                                                                    var _currentAddress = _address247;
                                                                                    _addr.save(_globalCurrentAddress, _address247);
                                                                                    return function () {
                                                                                        return timeit(globalStore, function (globalStore, result) {
                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                            return function () {
                                                                                                return Infer(globalStore, function (globalStore, _result17) {
                                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                    return function () {
                                                                                                        return expectation(globalStore, function (globalStore, _result16) {
                                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                            return function () {
                                                                                                                return round(globalStore, function (globalStore, _result10) {
                                                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                    return function () {
                                                                                                                        return Infer(globalStore, function (globalStore, _result13) {
                                                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                            return function () {
                                                                                                                                return expectation(globalStore, function (globalStore, _result12) {
                                                                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                                    return function () {
                                                                                                                                        return round(globalStore, function (globalStore, _result11) {
                                                                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                                            return function () {
                                                                                                                                                return _k9(globalStore, console.log(name, ':', _result10, ' ', _result11, ' ', result.runtimeInMilliseconds));
                                                                                                                                            };
                                                                                                                                        }, _address247.concat('_351'), _result12, 3);
                                                                                                                                    };
                                                                                                                                }, _address247.concat('_350'), _result13);
                                                                                                                            };
                                                                                                                        }, _address247.concat('_349'), {
                                                                                                                            model: function (globalStore, _k14, _address251) {
                                                                                                                                var _currentAddress = _address251;
                                                                                                                                _addr.save(_globalCurrentAddress, _address251);
                                                                                                                                return function () {
                                                                                                                                    return sample(globalStore, function (globalStore, _result15) {
                                                                                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                                        return function () {
                                                                                                                                            return _k14(globalStore, ad.scalar.sub(_result15.actions.length, 1));
                                                                                                                                        };
                                                                                                                                    }, _address251.concat('_348'), result.value);
                                                                                                                                };
                                                                                                                            }
                                                                                                                        });
                                                                                                                    };
                                                                                                                }, _address247.concat('_347'), _result16, 3);
                                                                                                            };
                                                                                                        }, _address247.concat('_346'), _result17);
                                                                                                    };
                                                                                                }, _address247.concat('_345'), {
                                                                                                    model: function (globalStore, _k18, _address250) {
                                                                                                        var _currentAddress = _address250;
                                                                                                        _addr.save(_globalCurrentAddress, _address250);
                                                                                                        return function () {
                                                                                                            return sample(globalStore, function (globalStore, _result19) {
                                                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                return function () {
                                                                                                                    return sum(globalStore, _k18, _address250.concat('_344'), _result19.rewards);
                                                                                                                };
                                                                                                            }, _address250.concat('_343'), result.value);
                                                                                                        };
                                                                                                    }
                                                                                                });
                                                                                            };
                                                                                        }, _address247.concat('_342'), function (globalStore, _k20, _address248) {
                                                                                            var _currentAddress = _address248;
                                                                                            _addr.save(_globalCurrentAddress, _address248);
                                                                                            return function () {
                                                                                                return Infer(globalStore, _k20, _address248.concat('_341'), {
                                                                                                    model: function (globalStore, _k21, _address249) {
                                                                                                        var _currentAddress = _address249;
                                                                                                        _addr.save(_globalCurrentAddress, _address249);
                                                                                                        return function () {
                                                                                                            return simulate(globalStore, function (globalStore, s) {
                                                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                return function () {
                                                                                                                    return _k21(globalStore, s);
                                                                                                                };
                                                                                                            }, _address249.concat('_340'), policy);
                                                                                                        };
                                                                                                    },
                                                                                                    method: 'forward',
                                                                                                    samples: 1000
                                                                                                });
                                                                                            };
                                                                                        });
                                                                                    };
                                                                                };
                                                                                var testParams = function testParams(globalStore, _k1, _address252, mu, sigma) {
                                                                                    var _currentAddress = _address252;
                                                                                    _addr.save(_globalCurrentAddress, _address252);
                                                                                    return function () {
                                                                                        return vals(globalStore, function (globalStore, _result7) {
                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                            return function () {
                                                                                                return probs(globalStore, function (globalStore, _result8) {
                                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                    return function () {
                                                                                                        return Categorical(globalStore, function (globalStore, _result6) {
                                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                            var _dummy5 = globalStore.reward = _result6;
                                                                                                            return function () {
                                                                                                                return enumPolicy(globalStore, function (globalStore, _result2) {
                                                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                    return function () {
                                                                                                                        return run(globalStore, _k1, _address252.concat('_357'), ad.scalar.add(ad.scalar.add(ad.scalar.add(ad.scalar.add('N(', mu), ', '), sigma), ')'), _result2);
                                                                                                                    };
                                                                                                                }, _address252.concat('_356'), {
                                                                                                                    myActions: function (globalStore, _k3, _address253, state) {
                                                                                                                        var _currentAddress = _address253;
                                                                                                                        _addr.save(_globalCurrentAddress, _address253);
                                                                                                                        return function () {
                                                                                                                            return actions(globalStore, function (globalStore, _result4) {
                                                                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                                return function () {
                                                                                                                                    return _k3(globalStore, L.firstUnobserved(_result4).concat([TERM_ACTION]));
                                                                                                                                };
                                                                                                                            }, _address253.concat('_355'), state);
                                                                                                                        };
                                                                                                                    }
                                                                                                                });
                                                                                                            };
                                                                                                        }, _address252.concat('_354'), {
                                                                                                            vs: _result7,
                                                                                                            ps: _result8
                                                                                                        });
                                                                                                    };
                                                                                                }, _address252.concat('_353'));
                                                                                            };
                                                                                        }, _address252.concat('_352'), mu, sigma);
                                                                                    };
                                                                                };
                                                                                return function () {
                                                                                    return testParams(globalStore, _k0, _address0.concat('_358'), ad.scalar.neg(2), 8);
                                                                                };
                                                                            }, _address0.concat('_339'), {
                                                                                vs: _result24,
                                                                                ps: _result25
                                                                            });
                                                                        };
                                                                    }, _address0.concat('_338'));
                                                                };
                                                            }, _address0.concat('_337'), 1, 2);
                                                        };
                                                    }, _address0.concat('_325'), function (globalStore, _k46, _address235, state) {
                                                        var _currentAddress = _address235;
                                                        _addr.save(_globalCurrentAddress, _address235);
                                                        return function () {
                                                            return stateTree(globalStore, function (globalStore, _result47) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return expectedObservationValue(globalStore, _k46, _address235.concat('_324'), _result47);
                                                                };
                                                            }, _address235.concat('_323'), state);
                                                        };
                                                    });
                                                };
                                            }, _address0.concat('_306'), 16, function (globalStore, _k70, _address226) {
                                                var _currentAddress = _address226;
                                                _addr.save(_globalCurrentAddress, _address226);
                                                return function () {
                                                    return _k70(globalStore, HIDDEN);
                                                };
                                            });
                                        };
                                    }, _address0.concat('_305'), {
                                        vs: [
                                            ad.scalar.neg(3),
                                            ad.scalar.neg(2),
                                            ad.scalar.neg(1),
                                            0,
                                            1,
                                            2,
                                            3
                                        ],
                                        ps: [
                                            0.006,
                                            0.061,
                                            0.242,
                                            0.383,
                                            0.242,
                                            0.061,
                                            0.006
                                        ]
                                    });
                                };
                            }, _address0.concat('_251'), {
                                vs: _result149,
                                ps: _result150
                            });
                        };
                    }, _address0.concat('_250'));
                };
            }, _address0.concat('_249'), 1, 2);
        };
    });
});

webppl.runEvaled(main, __runner__, {}, {}, topK, '');