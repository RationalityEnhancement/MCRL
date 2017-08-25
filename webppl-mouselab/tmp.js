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
        var Categorical = function Categorical(globalStore, _k545, _address7, params) {
            var _currentAddress = _address7;
            _addr.save(_globalCurrentAddress, _address7);
            return function () {
                return _k545(globalStore, util.jsnew(dists.Categorical, params));
            };
        };
        var idF = function idF(globalStore, _k407, _address71, x) {
            var _currentAddress = _address71;
            _addr.save(_globalCurrentAddress, _address71);
            return function () {
                return _k407(globalStore, x);
            };
        };
        var expectation = function expectation(globalStore, _k398, _address77, dist, func) {
            var _currentAddress = _address77;
            _addr.save(_globalCurrentAddress, _address77);
            var _k401 = function (globalStore, f) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                var supp = dist.support();
                return function () {
                    return reduce(globalStore, _k398, _address77.concat('_69'), function (globalStore, _k399, _address78, s, acc) {
                        var _currentAddress = _address78;
                        _addr.save(_globalCurrentAddress, _address78);
                        return function () {
                            return f(globalStore, function (globalStore, _result400) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return _k399(globalStore, ad.scalar.add(acc, ad.scalar.mul(ad.scalar.exp(dist.score(s)), _result400)));
                                };
                            }, _address78.concat('_68'), s);
                        };
                    }, 0, supp);
                };
            };
            return function () {
                return func ? _k401(globalStore, func) : _k401(globalStore, idF);
            };
        };
        var map_helper = function map_helper(globalStore, _k382, _address91, i, j, f) {
            var _currentAddress = _address91;
            _addr.save(_globalCurrentAddress, _address91);
            var n = ad.scalar.add(ad.scalar.sub(j, i), 1);
            return function () {
                return ad.scalar.eq(n, 0) ? _k382(globalStore, []) : ad.scalar.eq(n, 1) ? f(globalStore, function (globalStore, _result383) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return _k382(globalStore, [_result383]);
                    };
                }, _address91.concat('_70'), i) : function (globalStore, n1) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return map_helper(globalStore, function (globalStore, _result384) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return map_helper(globalStore, function (globalStore, _result385) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return _k382(globalStore, _result384.concat(_result385));
                                    };
                                }, _address91.concat('_72'), ad.scalar.add(i, n1), j, f);
                            };
                        }, _address91.concat('_71'), i, ad.scalar.sub(ad.scalar.add(i, n1), 1), f);
                    };
                }(globalStore, ad.scalar.ceil(ad.scalar.div(n, 2)));
            };
        };
        var map = function map(globalStore, _k380, _address92, fn, l) {
            var _currentAddress = _address92;
            _addr.save(_globalCurrentAddress, _address92);
            return function () {
                return map_helper(globalStore, _k380, _address92.concat('_74'), 0, ad.scalar.sub(l.length, 1), function (globalStore, _k381, _address93, i) {
                    var _currentAddress = _address93;
                    _addr.save(_globalCurrentAddress, _address93);
                    return function () {
                        return fn(globalStore, _k381, _address93.concat('_73'), l[i]);
                    };
                });
            };
        };
        var map2 = function map2(globalStore, _k378, _address94, fn, l1, l2) {
            var _currentAddress = _address94;
            _addr.save(_globalCurrentAddress, _address94);
            return function () {
                return map_helper(globalStore, _k378, _address94.concat('_76'), 0, ad.scalar.sub(l1.length, 1), function (globalStore, _k379, _address95, i) {
                    var _currentAddress = _address95;
                    _addr.save(_globalCurrentAddress, _address95);
                    return function () {
                        return fn(globalStore, _k379, _address95.concat('_75'), l1[i], l2[i]);
                    };
                });
            };
        };
        var mapIndexed = function mapIndexed(globalStore, _k374, _address98, fn, l) {
            var _currentAddress = _address98;
            _addr.save(_globalCurrentAddress, _address98);
            return function () {
                return map_helper(globalStore, _k374, _address98.concat('_80'), 0, ad.scalar.sub(l.length, 1), function (globalStore, _k375, _address99, i) {
                    var _currentAddress = _address99;
                    _addr.save(_globalCurrentAddress, _address99);
                    return function () {
                        return fn(globalStore, _k375, _address99.concat('_79'), i, l[i]);
                    };
                });
            };
        };
        var extend = function extend(globalStore, _k369, _address102) {
            var _currentAddress = _address102;
            _addr.save(_globalCurrentAddress, _address102);
            var _arguments2 = Array.prototype.slice.call(arguments, 3);
            return function () {
                return _k369(globalStore, _.assign.apply(_, [{}].concat(_arguments2)));
            };
        };
        var reduce = function reduce(globalStore, _k366, _address103, fn, init, ar) {
            var _currentAddress = _address103;
            _addr.save(_globalCurrentAddress, _address103);
            var n = ar.length;
            var helper = function helper(globalStore, _k367, _address104, i) {
                var _currentAddress = _address104;
                _addr.save(_globalCurrentAddress, _address104);
                return function () {
                    return ad.scalar.peq(i, n) ? _k367(globalStore, init) : helper(globalStore, function (globalStore, _result368) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return fn(globalStore, _k367, _address104.concat('_84'), ar[i], _result368);
                        };
                    }, _address104.concat('_83'), ad.scalar.add(i, 1));
                };
            };
            return function () {
                return helper(globalStore, _k366, _address103.concat('_85'), 0);
            };
        };
        var sum = function sum(globalStore, _k364, _address105, l) {
            var _currentAddress = _address105;
            _addr.save(_globalCurrentAddress, _address105);
            return function () {
                return reduce(globalStore, _k364, _address105.concat('_86'), function (globalStore, _k365, _address106, a, b) {
                    var _currentAddress = _address106;
                    _addr.save(_globalCurrentAddress, _address106);
                    return function () {
                        return _k365(globalStore, ad.scalar.add(a, b));
                    };
                }, 0, l);
            };
        };
        var zip = function zip(globalStore, _k346, _address119, xs, ys) {
            var _currentAddress = _address119;
            _addr.save(_globalCurrentAddress, _address119);
            return function () {
                return map2(globalStore, _k346, _address119.concat('_98'), function (globalStore, _k347, _address120, x, y) {
                    var _currentAddress = _address120;
                    _addr.save(_globalCurrentAddress, _address120);
                    return function () {
                        return _k347(globalStore, [
                            x,
                            y
                        ]);
                    };
                }, xs, ys);
            };
        };
        var filter = function filter(globalStore, _k341, _address121, fn, ar) {
            var _currentAddress = _address121;
            _addr.save(_globalCurrentAddress, _address121);
            var helper = function helper(globalStore, _k342, _address122, i, j) {
                var _currentAddress = _address122;
                _addr.save(_globalCurrentAddress, _address122);
                var n = ad.scalar.add(ad.scalar.sub(j, i), 1);
                return function () {
                    return ad.scalar.eq(n, 0) ? _k342(globalStore, []) : ad.scalar.eq(n, 1) ? fn(globalStore, function (globalStore, _result343) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return _result343 ? _k342(globalStore, [ar[i]]) : _k342(globalStore, []);
                        };
                    }, _address122.concat('_99'), ar[i]) : function (globalStore, n1) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return helper(globalStore, function (globalStore, _result344) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return helper(globalStore, function (globalStore, _result345) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        return function () {
                                            return _k342(globalStore, _result344.concat(_result345));
                                        };
                                    }, _address122.concat('_101'), ad.scalar.add(i, n1), j);
                                };
                            }, _address122.concat('_100'), i, ad.scalar.sub(ad.scalar.add(i, n1), 1));
                        };
                    }(globalStore, ad.scalar.ceil(ad.scalar.div(n, 2)));
                };
            };
            return function () {
                return helper(globalStore, _k341, _address121.concat('_102'), 0, ad.scalar.sub(ar.length, 1));
            };
        };
        var maxWith = function maxWith(globalStore, _k328, _address129, f, ar) {
            var _currentAddress = _address129;
            _addr.save(_globalCurrentAddress, _address129);
            var fn = function fn(globalStore, _k331, _address130, _ar, _best) {
                var _currentAddress = _address130;
                _addr.save(_globalCurrentAddress, _address130);
                return function () {
                    return ad.scalar.peq(_ar.length, 0) ? _k331(globalStore, _best) : ad.scalar.gt(_ar[0][1], _best[1]) ? fn(globalStore, _k331, _address130.concat('_112'), _ar.slice(1), _ar[0]) : fn(globalStore, _k331, _address130.concat('_113'), _ar.slice(1), _best);
                };
            };
            return function () {
                return map(globalStore, function (globalStore, _result330) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return zip(globalStore, function (globalStore, _result329) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return fn(globalStore, _k328, _address129.concat('_116'), _result329, [
                                    ad.scalar.neg(Infinity),
                                    ad.scalar.neg(Infinity)
                                ]);
                            };
                        }, _address129.concat('_115'), ar, _result330);
                    };
                }, _address129.concat('_114'), f, ar);
            };
        };
        var error = function error(globalStore, _k278, _address147, msg) {
            var _currentAddress = _address147;
            _addr.save(_globalCurrentAddress, _address147);
            return function () {
                return _k278(globalStore, util.error(msg));
            };
        };
        var SampleGuide = function SampleGuide(globalStore, _k274, _address151, wpplFn, options) {
            var _currentAddress = _address151;
            _addr.save(_globalCurrentAddress, _address151);
            return function () {
                return ForwardSample(globalStore, _k274, _address151.concat('_152'), wpplFn, _.assign({ guide: !0 }, _.omit(options, 'guide')));
            };
        };
        var OptimizeThenSample = function OptimizeThenSample(globalStore, _k272, _address152, wpplFn, options) {
            var _currentAddress = _address152;
            _addr.save(_globalCurrentAddress, _address152);
            return function () {
                return Optimize(globalStore, function (globalStore, _dummy273) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var opts = _.pick(options, 'samples', 'onlyMAP', 'verbose');
                    return function () {
                        return SampleGuide(globalStore, _k272, _address152.concat('_154'), wpplFn, opts);
                    };
                }, _address152.concat('_153'), wpplFn, _.omit(options, 'samples', 'onlyMAP'));
            };
        };
        var DefaultInfer = function DefaultInfer(globalStore, _k262, _address153, wpplFn, options) {
            var _currentAddress = _address153;
            _addr.save(_globalCurrentAddress, _address153);
            var _dummy271 = util.mergeDefaults(options, {}, 'Infer');
            var maxEnumTreeSize = 200000;
            var minSampleRate = 250;
            var samples = 1000;
            return function () {
                return Enumerate(globalStore, function (globalStore, enumResult) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var _k270 = function (globalStore, _dummy269) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        var _dummy268 = console.log('Using "rejection"');
                        return function () {
                            return Rejection(globalStore, function (globalStore, rejResult) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return rejResult instanceof Error ? function (globalStore, _dummy267) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        return function () {
                                            return CheckSampleAfterFactor(globalStore, function (globalStore, hasSampleAfterFactor) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                var _k265 = function (globalStore, _dummy264) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    var _dummy263 = console.log('Using "MCMC"');
                                                    return function () {
                                                        return MCMC(globalStore, _k262, _address153.concat('_161'), wpplFn, { samples: samples });
                                                    };
                                                };
                                                return function () {
                                                    return hasSampleAfterFactor ? function (globalStore, _dummy266) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return SMC(globalStore, function (globalStore, smcResult) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return dists.isDist(smcResult) ? _k262(globalStore, smcResult) : smcResult instanceof Error ? _k265(globalStore, console.log(ad.scalar.add(smcResult.message, '..quit SMC'))) : error(globalStore, _k265, _address153.concat('_160'), 'Invalid return value from SMC');
                                                                };
                                                            }, _address153.concat('_159'), wpplFn, {
                                                                throwOnError: !1,
                                                                particles: samples
                                                            });
                                                        };
                                                    }(globalStore, console.log('Using "SMC" (interleaving samples and factors detected)')) : _k265(globalStore, undefined);
                                                };
                                            }, _address153.concat('_158'), wpplFn);
                                        };
                                    }(globalStore, console.log(ad.scalar.add(rejResult.message, '..quit rejection'))) : dists.isDist(rejResult) ? _k262(globalStore, rejResult) : error(globalStore, _k262, _address153.concat('_162'), 'Invalid return value from rejection');
                                };
                            }, _address153.concat('_157'), wpplFn, {
                                minSampleRate: minSampleRate,
                                throwOnError: !1,
                                samples: samples
                            });
                        };
                    };
                    return function () {
                        return dists.isDist(enumResult) ? _k262(globalStore, enumResult) : enumResult instanceof Error ? _k270(globalStore, console.log(ad.scalar.add(enumResult.message, '..quit enumerate'))) : error(globalStore, _k270, _address153.concat('_156'), 'Invalid return value from enumerate');
                    };
                }, _address153.concat('_155'), wpplFn, {
                    maxEnumTreeSize: maxEnumTreeSize,
                    maxRuntimeInMS: 5000,
                    throwOnError: !1,
                    strategy: 'depthFirst'
                });
            };
        };
        var Infer = function Infer(globalStore, _k255, _address154, options, maybeFn) {
            var _currentAddress = _address154;
            _addr.save(_globalCurrentAddress, _address154);
            var _k261 = function (globalStore, wpplFn) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                var _k260 = function (globalStore, _dummy259) {
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
                    var _k258 = function (globalStore, methodName) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        var _k257 = function (globalStore, _dummy256) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            var method = methodMap[methodName];
                            return function () {
                                return method(globalStore, _k255, _address154.concat('_165'), wpplFn, _.omit(options, 'method', 'model'));
                            };
                        };
                        return function () {
                            return _.has(methodMap, methodName) ? _k257(globalStore, undefined) : function (globalStore, methodNames) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                var msg = ad.scalar.add(ad.scalar.add(ad.scalar.add(ad.scalar.add('Infer: \'', methodName), '\' is not a valid method. The following methods are available: '), methodNames.join(', ')), '.');
                                return function () {
                                    return error(globalStore, _k257, _address154.concat('_164'), msg);
                                };
                            }(globalStore, _.keys(methodMap));
                        };
                    };
                    return function () {
                        return options.method ? _k258(globalStore, options.method) : _k258(globalStore, 'defaultInfer');
                    };
                };
                return function () {
                    return _.isFunction(wpplFn) ? _k260(globalStore, undefined) : error(globalStore, _k260, _address154.concat('_163'), 'Infer: a model was not specified.');
                };
            };
            return function () {
                return util.isObject(options) ? maybeFn ? _k261(globalStore, maybeFn) : _k261(globalStore, options.model) : _k261(globalStore, options);
            };
        };
        var utils = webpplMouselab;
        var TERM_ACTION = '__TERM_ACTION__';
        var TERM_STATE = '__TERM_STATE__';
        var UNKNOWN = '__';
        var INITIAL_NODE = 0;
        var env = utils.buildEnv();
        var tree = env.tree;
        var nodes = _.range(tree.length);
        var children = function children(globalStore, _k139, _address188, node) {
            var _currentAddress = _address188;
            _addr.save(_globalCurrentAddress, _address188);
            return function () {
                return _k139(globalStore, tree[node]);
            };
        };
        var nodeReward = function nodeReward(globalStore, _k138, _address189, node) {
            var _currentAddress = _address189;
            _addr.save(_globalCurrentAddress, _address189);
            return function () {
                return ad.scalar.eq(node, INITIAL_NODE) ? _k138(globalStore, 0) : sample(globalStore, _k138, _address189.concat('_248'), globalStore.reward);
            };
        };
        var expectedNodeReward = function expectedNodeReward(globalStore, _k137, _address190, state, node) {
            var _currentAddress = _address190;
            _addr.save(_globalCurrentAddress, _address190);
            return function () {
                return ad.scalar.eq(node, INITIAL_NODE) ? _k137(globalStore, 0) : ad.scalar.eq(state[node], UNKNOWN) ? expectation(globalStore, _k137, _address190.concat('_249'), globalStore.reward) : _k137(globalStore, state[node]);
            };
        };
        var nodeQuality = dp.cache(function (globalStore, _k130, _address191, state, node) {
            var _currentAddress = _address191;
            _addr.save(_globalCurrentAddress, _address191);
            return function () {
                return children(globalStore, function (globalStore, _result132) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var _k133 = function (globalStore, best_child_val) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return expectedNodeReward(globalStore, function (globalStore, _result131) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return _k130(globalStore, ad.scalar.add(_result131, best_child_val));
                                };
                            }, _address191.concat('_254'), state, node);
                        };
                    };
                    return function () {
                        return ad.scalar.eq(_result132.length, 0) ? _k133(globalStore, 0) : children(globalStore, function (globalStore, _result136) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return map(globalStore, function (globalStore, _result134) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return _k133(globalStore, _.max(_result134));
                                    };
                                }, _address191.concat('_253'), function (globalStore, _k135, _address192, child) {
                                    var _currentAddress = _address192;
                                    _addr.save(_globalCurrentAddress, _address192);
                                    return function () {
                                        return nodeQuality(globalStore, _k135, _address192.concat('_251'), state, child);
                                    };
                                }, _result136);
                            };
                        }, _address191.concat('_252'), node);
                    };
                }, _address191.concat('_250'), node);
            };
        });
        var _dummy129 = globalStore.energySpent = 1;
        var _dummy128 = globalStore.rewardAccrued = 0;
        var termReward = function termReward(globalStore, _k127, _address193, state) {
            var _currentAddress = _address193;
            _addr.save(_globalCurrentAddress, _address193);
            return function () {
                return nodeQuality(globalStore, function (globalStore, expectedReward) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return _k127(globalStore, expectedReward);
                    };
                }, _address193.concat('_255'), state, INITIAL_NODE);
            };
        };
        var transition = function transition(globalStore, _k117, _address194, state, action) {
            var _currentAddress = _address194;
            _addr.save(_globalCurrentAddress, _address194);
            var _k126 = function (globalStore, _dummy125) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                var _k124 = function (globalStore, _dummy123) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var _k120 = function (globalStore, _dummy119) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return nodeReward(globalStore, function (globalStore, _result118) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return _k117(globalStore, utils.updateList(state, action, _result118));
                                };
                            }, _address194.concat('_259'), action);
                        };
                    };
                    return function () {
                        return ad.scalar.neq(state[action], UNKNOWN) ? actions(globalStore, function (globalStore, _result122) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            var _dummy121 = console.log(_result122);
                            return function () {
                                return error(globalStore, _k120, _address194.concat('_258'), ad.scalar.add(ad.scalar.add(ad.scalar.add('observing state twice\n', JSON.stringify(state)), ' '), action));
                            };
                        }, _address194.concat('_257'), state) : _k120(globalStore, undefined);
                    };
                };
                return function () {
                    return ad.scalar.eq(action, TERM_ACTION) ? _k117(globalStore, TERM_STATE) : _k124(globalStore, undefined);
                };
            };
            return function () {
                return ad.scalar.eq(state, TERM_STATE) ? error(globalStore, _k126, _address194.concat('_256'), ad.scalar.add('transition from term ', action)) : _k126(globalStore, undefined);
            };
        };
        var reward = function reward(globalStore, _k116, _address195, state, action) {
            var _currentAddress = _address195;
            _addr.save(_globalCurrentAddress, _address195);
            return function () {
                return ad.scalar.eq(action, TERM_ACTION) ? termReward(globalStore, _k116, _address195.concat('_260'), state) : _k116(globalStore, globalStore.cost);
            };
        };
        var unobservedNodes = function unobservedNodes(globalStore, _k114, _address196, state) {
            var _currentAddress = _address196;
            _addr.save(_globalCurrentAddress, _address196);
            return function () {
                return filter(globalStore, _k114, _address196.concat('_261'), function (globalStore, _k115, _address197, node) {
                    var _currentAddress = _address197;
                    _addr.save(_globalCurrentAddress, _address197);
                    return function () {
                        return _k115(globalStore, ad.scalar.eq(state[node], UNKNOWN));
                    };
                }, nodes);
            };
        };
        var actions = function actions(globalStore, _k109, _address200, state) {
            var _currentAddress = _address200;
            _addr.save(_globalCurrentAddress, _address200);
            return function () {
                return unobservedNodes(globalStore, function (globalStore, _result110) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    return function () {
                        return _k109(globalStore, _result110.concat([TERM_ACTION]));
                    };
                }, _address200.concat('_263'), state);
            };
        };
        var enumPolicy = function enumPolicy(globalStore, _k82, _address207, opts) {
            var _currentAddress = _address207;
            _addr.save(_globalCurrentAddress, _address207);
            return function () {
                return extend(globalStore, function (globalStore, params) {
                    _addr.save(_globalCurrentAddress, _currentAddress);
                    var myActions = params.myActions;
                    var actionAndValue = dp.cache(function (globalStore, _k87, _address208, state) {
                        var _currentAddress = _address208;
                        _addr.save(_globalCurrentAddress, _address208);
                        var Q = function Q(globalStore, _k95, _address209, action) {
                            var _currentAddress = _address209;
                            _addr.save(_globalCurrentAddress, _address209);
                            return function () {
                                return Infer(globalStore, function (globalStore, _result96) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return expectation(globalStore, _k95, _address209.concat('_276'), _result96);
                                    };
                                }, _address209.concat('_275'), {
                                    model: function (globalStore, _k97, _address210) {
                                        var _currentAddress = _address210;
                                        _addr.save(_globalCurrentAddress, _address210);
                                        return function () {
                                            return transition(globalStore, function (globalStore, newState) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return reward(globalStore, function (globalStore, _result98) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return V(globalStore, function (globalStore, _result99) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return _k97(globalStore, ad.scalar.add(_result98, _result99));
                                                                };
                                                            }, _address210.concat('_274'), newState);
                                                        };
                                                    }, _address210.concat('_273'), state, action);
                                                };
                                            }, _address210.concat('_272'), state, action);
                                        };
                                    },
                                    method: 'enumerate'
                                });
                            };
                        };
                        var _k93 = function (globalStore, _dummy92) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return myActions(globalStore, function (globalStore, _result91) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return maxWith(globalStore, function (globalStore, result) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            var _k89 = function (globalStore, _dummy88) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return _k87(globalStore, result);
                                                };
                                            };
                                            return function () {
                                                return ad.scalar.eq(result[0], ad.scalar.neg(Infinity)) ? myActions(globalStore, function (globalStore, _result90) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    return function () {
                                                        return error(globalStore, _k89, _address208.concat('_282'), ad.scalar.add('problem!\n', _result90));
                                                    };
                                                }, _address208.concat('_281'), state) : _k89(globalStore, undefined);
                                            };
                                        }, _address208.concat('_280'), Q, _result91);
                                    };
                                }, _address208.concat('_279'), state);
                            };
                        };
                        return function () {
                            return myActions(globalStore, function (globalStore, _result94) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return ad.scalar.eq(_result94.length, 0) ? error(globalStore, _k93, _address208.concat('_278'), 'no actions') : _k93(globalStore, undefined);
                                };
                            }, _address208.concat('_277'), state);
                        };
                    });
                    var V = utils.cache(function (globalStore, _k85, _address211, state) {
                        var _currentAddress = _address211;
                        _addr.save(_globalCurrentAddress, _address211);
                        return function () {
                            return ad.scalar.eq(state, TERM_STATE) ? _k85(globalStore, 0) : actionAndValue(globalStore, function (globalStore, _result86) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return _k85(globalStore, _result86[1]);
                                };
                            }, _address211.concat('_283'), state);
                        };
                    }, env);
                    var policy = function policy(globalStore, _k83, _address212, state) {
                        var _currentAddress = _address212;
                        _addr.save(_globalCurrentAddress, _address212);
                        return function () {
                            return actionAndValue(globalStore, function (globalStore, _result84) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                var a = _result84[0];
                                return function () {
                                    return _k83(globalStore, a);
                                };
                            }, _address212.concat('_284'), state);
                        };
                    };
                    return function () {
                        return _k82(globalStore, policy);
                    };
                }, _address207.concat('_271'), {
                    maxExecutions: Infinity,
                    alpha: 1000,
                    myActions: actions
                }, opts);
            };
        };
        var simulate = function simulate(globalStore, _k80, _address213, policy) {
            var _currentAddress = _address213;
            _addr.save(_globalCurrentAddress, _address213);
            var rec = function rec(globalStore, _k81, _address214, acc) {
                var _currentAddress = _address214;
                _addr.save(_globalCurrentAddress, _address214);
                var state = _.last(acc.states);
                return function () {
                    return ad.scalar.eq(state, TERM_STATE) ? _k81(globalStore, acc) : policy(globalStore, function (globalStore, action) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        return function () {
                            return transition(globalStore, function (globalStore, newState) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return reward(globalStore, function (globalStore, r) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        return function () {
                                            return rec(globalStore, _k81, _address214.concat('_288'), {
                                                states: acc.states.concat([newState]),
                                                rewards: acc.rewards.concat([r]),
                                                actions: acc.actions.concat([action])
                                            });
                                        };
                                    }, _address214.concat('_287'), state, action);
                                };
                            }, _address214.concat('_286'), state, action);
                        };
                    }, _address214.concat('_285'), state);
                };
            };
            return function () {
                return rec(globalStore, _k80, _address213.concat('_289'), {
                    states: [env.initialState],
                    rewards: [],
                    actions: []
                });
            };
        };
        return function () {
            return Categorical(globalStore, function (globalStore, REWARD) {
                _addr.save(_globalCurrentAddress, _currentAddress);
                var COST = ad.scalar.neg(1);
                var value = function value(globalStore, _k79, _address215, tree) {
                    var _currentAddress = _address215;
                    _addr.save(_globalCurrentAddress, _address215);
                    return function () {
                        return _k79(globalStore, tree[0]);
                    };
                };
                var children = function children(globalStore, _k78, _address216, tree) {
                    var _currentAddress = _address216;
                    _addr.save(_globalCurrentAddress, _address216);
                    return function () {
                        return _k78(globalStore, tree[1]);
                    };
                };
                var OBSERVE = '__OBSERVE__';
                var subjectiveReward = function subjectiveReward(globalStore, _k77, _address217, tree) {
                    var _currentAddress = _address217;
                    _addr.save(_globalCurrentAddress, _address217);
                    return function () {
                        return value(globalStore, function (globalStore, v) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return ad.scalar.eq(v, UNKNOWN) ? expectation(globalStore, _k77, _address217.concat('_292'), globalStore.reward) : ad.scalar.eq(v, OBSERVE) ? sample(globalStore, _k77, _address217.concat('_293'), globalStore.reward) : _k77(globalStore, v);
                            };
                        }, _address217.concat('_291'), tree);
                    };
                };
                var observationValue = dp.cache(function (globalStore, _k67, _address218, tree, params) {
                    var _currentAddress = _address218;
                    _addr.save(_globalCurrentAddress, _address218);
                    return function () {
                        return extend(globalStore, function (globalStore, params) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return extend(globalStore, function (globalStore, _result68) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return Infer(globalStore, _k67, _address218.concat('_302'), _result68);
                                    };
                                }, _address218.concat('_301'), params, {
                                    model: function (globalStore, _k69, _address219) {
                                        var _currentAddress = _address219;
                                        _addr.save(_globalCurrentAddress, _address219);
                                        return function () {
                                            return children(globalStore, function (globalStore, _result71) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                var _k72 = function (globalStore, bestChildVal) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    return function () {
                                                        return subjectiveReward(globalStore, function (globalStore, _result70) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return _k69(globalStore, ad.scalar.add(_result70, bestChildVal));
                                                            };
                                                        }, _address219.concat('_300'), tree);
                                                    };
                                                };
                                                return function () {
                                                    return ad.scalar.eq(_result71.length, 0) ? _k72(globalStore, 0) : children(globalStore, function (globalStore, _result76) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return map(globalStore, function (globalStore, _result73) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return _k72(globalStore, _.max(_result73));
                                                                };
                                                            }, _address219.concat('_299'), function (globalStore, _k74, _address220, child) {
                                                                var _currentAddress = _address220;
                                                                _addr.save(_globalCurrentAddress, _address220);
                                                                return function () {
                                                                    return observationValue(globalStore, function (globalStore, _result75) {
                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                        return function () {
                                                                            return sample(globalStore, _k74, _address220.concat('_297'), _result75);
                                                                        };
                                                                    }, _address220.concat('_296'), child, params);
                                                                };
                                                            }, _result76);
                                                        };
                                                    }, _address219.concat('_298'), tree);
                                                };
                                            }, _address219.concat('_295'), tree);
                                        };
                                    }
                                });
                            };
                        }, _address218.concat('_294'), { method: 'enumerate' }, params);
                    };
                });
                var expectedObservationValue = function expectedObservationValue(globalStore, _k65, _address221, tree, params) {
                    var _currentAddress = _address221;
                    _addr.save(_globalCurrentAddress, _address221);
                    var startTime = Date.now();
                    return function () {
                        return observationValue(globalStore, function (globalStore, _result66) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return expectation(globalStore, function (globalStore, result) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return _k65(globalStore, result);
                                    };
                                }, _address221.concat('_304'), _result66);
                            };
                        }, _address221.concat('_303'), tree, params);
                    };
                };
                var stateTree = function stateTree(globalStore, _k64, _address222, state) {
                    var _currentAddress = _address222;
                    _addr.save(_globalCurrentAddress, _address222);
                    var s = state;
                    return function () {
                        return _k64(globalStore, [
                            s[0],
                            [
                                [
                                    s[1],
                                    [[
                                            s[2],
                                            [
                                                [
                                                    s[3],
                                                    []
                                                ],
                                                [
                                                    s[4],
                                                    []
                                                ]
                                            ]
                                        ]]
                                ],
                                [
                                    s[5],
                                    [[
                                            s[6],
                                            [
                                                [
                                                    s[7],
                                                    []
                                                ],
                                                [
                                                    s[8],
                                                    []
                                                ]
                                            ]
                                        ]]
                                ],
                                [
                                    s[9],
                                    [[
                                            s[10],
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
                                    s[13],
                                    [[
                                            s[14],
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
                var actionGroups = [
                    [
                        1,
                        2,
                        3,
                        4
                    ],
                    [
                        1,
                        2,
                        3,
                        4
                    ],
                    [
                        1,
                        2,
                        3,
                        4
                    ],
                    [
                        1,
                        2,
                        3,
                        4
                    ],
                    [
                        1,
                        2,
                        3,
                        4
                    ],
                    [
                        5,
                        6,
                        7,
                        8
                    ],
                    [
                        5,
                        6,
                        7,
                        8
                    ],
                    [
                        5,
                        6,
                        7,
                        8
                    ],
                    [
                        5,
                        6,
                        7,
                        8
                    ],
                    [
                        5,
                        6,
                        7,
                        8
                    ],
                    [
                        9,
                        10,
                        11,
                        12
                    ],
                    [
                        9,
                        10,
                        11,
                        12
                    ],
                    [
                        9,
                        10,
                        11,
                        12
                    ],
                    [
                        9,
                        10,
                        11,
                        12
                    ],
                    [
                        9,
                        10,
                        11,
                        12
                    ],
                    [
                        13,
                        14,
                        15,
                        16
                    ],
                    [
                        13,
                        14,
                        15,
                        16
                    ],
                    [
                        13,
                        14,
                        15,
                        16
                    ],
                    [
                        13,
                        14,
                        15,
                        16
                    ],
                    [
                        13,
                        14,
                        15,
                        16
                    ]
                ];
                var obsTree = function obsTree(globalStore, _k59, _address223, state, toObserve) {
                    var _currentAddress = _address223;
                    _addr.save(_globalCurrentAddress, _address223);
                    return function () {
                        return mapIndexed(globalStore, function (globalStore, _result60) {
                            _addr.save(_globalCurrentAddress, _currentAddress);
                            return function () {
                                return stateTree(globalStore, _k59, _address223.concat('_306'), _result60);
                            };
                        }, _address223.concat('_305'), function (globalStore, _k61, _address224, i, r) {
                            var _currentAddress = _address224;
                            _addr.save(_globalCurrentAddress, _address224);
                            var _k63 = function (globalStore, _result62) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return _result62 ? _k61(globalStore, OBSERVE) : _k61(globalStore, r);
                                };
                            };
                            return function () {
                                return _.includes(toObserve, i) ? _k63(globalStore, ad.scalar.eq(r, UNKNOWN)) : _k63(globalStore, _.includes(toObserve, i));
                            };
                        }, state);
                    };
                };
                return function () {
                    return cache(globalStore, function (globalStore, termValue) {
                        _addr.save(_globalCurrentAddress, _currentAddress);
                        var VOC_1 = function VOC_1(globalStore, _k53, _address226, state, action) {
                            var _currentAddress = _address226;
                            _addr.save(_globalCurrentAddress, _address226);
                            return function () {
                                return obsTree(globalStore, function (globalStore, _result56) {
                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                    return function () {
                                        return expectedObservationValue(globalStore, function (globalStore, _result54) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            return function () {
                                                return termValue(globalStore, function (globalStore, _result55) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    return function () {
                                                        return _k53(globalStore, ad.scalar.sub(_result54, _result55));
                                                    };
                                                }, _address226.concat('_312'), state);
                                            };
                                        }, _address226.concat('_311'), _result56);
                                    };
                                }, _address226.concat('_310'), state, [action]);
                            };
                        };
                        return function () {
                            return cache(globalStore, function (globalStore, VPI_full) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                var VPI_action = function VPI_action(globalStore, _k45, _address228, state, action) {
                                    var _currentAddress = _address228;
                                    _addr.save(_globalCurrentAddress, _address228);
                                    var obs = actionGroups[action];
                                    return function () {
                                        return obsTree(globalStore, function (globalStore, _result48) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            return function () {
                                                return expectedObservationValue(globalStore, function (globalStore, _result46) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    return function () {
                                                        return termValue(globalStore, function (globalStore, _result47) {
                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                            return function () {
                                                                return _k45(globalStore, ad.scalar.sub(_result46, _result47));
                                                            };
                                                        }, _address228.concat('_319'), state);
                                                    };
                                                }, _address228.concat('_318'), _result48);
                                            };
                                        }, _address228.concat('_317'), state, obs);
                                    };
                                };
                                var dot = function dot(globalStore, _k42, _address229, x, y) {
                                    var _currentAddress = _address229;
                                    _addr.save(_globalCurrentAddress, _address229);
                                    return function () {
                                        return map2(globalStore, function (globalStore, _result43) {
                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                            return function () {
                                                return sum(globalStore, _k42, _address229.concat('_321'), _result43);
                                            };
                                        }, _address229.concat('_320'), function (globalStore, _k44, _address230, x, y) {
                                            var _currentAddress = _address230;
                                            _addr.save(_globalCurrentAddress, _address230);
                                            return function () {
                                                return _k44(globalStore, ad.scalar.mul(x, y));
                                            };
                                        }, x, y);
                                    };
                                };
                                return function () {
                                    return cache(globalStore, function (globalStore, Q_meta) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        var timeit = function timeit(globalStore, _k30, _address235, thunk) {
                                            var _currentAddress = _address235;
                                            _addr.save(_globalCurrentAddress, _address235);
                                            var t0 = webpplTimeit.now();
                                            return function () {
                                                return thunk(globalStore, function (globalStore, value) {
                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                    var t1 = webpplTimeit.now();
                                                    return function () {
                                                        return _k30(globalStore, {
                                                            value: value,
                                                            runtimeInMilliseconds: ad.scalar.sub(t1, t0)
                                                        });
                                                    };
                                                }, _address235.concat('_333'));
                                            };
                                        };
                                        var L = webpplMouselab;
                                        var round = function round(globalStore, _k28, _address237, x, p) {
                                            var _currentAddress = _address237;
                                            _addr.save(_globalCurrentAddress, _address237);
                                            return function () {
                                                return _k28(globalStore, ad.scalar.div(ad.scalar.round(ad.scalar.mul(x, ad.scalar.pow(10, p))), ad.scalar.pow(10, p)));
                                            };
                                        };
                                        var vals = function vals(globalStore, _k27, _address238, mu, sigma) {
                                            var _currentAddress = _address238;
                                            _addr.save(_globalCurrentAddress, _address238);
                                            return function () {
                                                return _k27(globalStore, [
                                                    ad.scalar.neg(15),
                                                    ad.scalar.neg(5),
                                                    5,
                                                    15
                                                ]);
                                            };
                                        };
                                        var probs = function probs(globalStore, _k26, _address239) {
                                            var _currentAddress = _address239;
                                            _addr.save(_globalCurrentAddress, _address239);
                                            return function () {
                                                return _k26(globalStore, [
                                                    0.159,
                                                    0.341,
                                                    0.341,
                                                    0.159
                                                ]);
                                            };
                                        };
                                        var _dummy25 = globalStore.cost = ad.scalar.neg(1);
                                        return function () {
                                            return vals(globalStore, function (globalStore, _result23) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return probs(globalStore, function (globalStore, _result24) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return Categorical(globalStore, function (globalStore, _result22) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                var _dummy21 = globalStore.reward = _result22;
                                                                var run = function run(globalStore, _k8, _address240, name, policy) {
                                                                    var _currentAddress = _address240;
                                                                    _addr.save(_globalCurrentAddress, _address240);
                                                                    return function () {
                                                                        return timeit(globalStore, function (globalStore, result) {
                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                            return function () {
                                                                                return Infer(globalStore, function (globalStore, _result16) {
                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                    return function () {
                                                                                        return expectation(globalStore, function (globalStore, _result15) {
                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                            return function () {
                                                                                                return round(globalStore, function (globalStore, _result9) {
                                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                    return function () {
                                                                                                        return Infer(globalStore, function (globalStore, _result12) {
                                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                            return function () {
                                                                                                                return expectation(globalStore, function (globalStore, _result11) {
                                                                                                                    _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                    return function () {
                                                                                                                        return round(globalStore, function (globalStore, _result10) {
                                                                                                                            _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                            return function () {
                                                                                                                                return _k8(globalStore, console.log(name, ':', _result9, ' ', _result10, ' ', result.runtimeInMilliseconds));
                                                                                                                            };
                                                                                                                        }, _address240.concat('_348'), _result11, 3);
                                                                                                                    };
                                                                                                                }, _address240.concat('_347'), _result12);
                                                                                                            };
                                                                                                        }, _address240.concat('_346'), {
                                                                                                            model: function (globalStore, _k13, _address244) {
                                                                                                                var _currentAddress = _address244;
                                                                                                                _addr.save(_globalCurrentAddress, _address244);
                                                                                                                return function () {
                                                                                                                    return sample(globalStore, function (globalStore, _result14) {
                                                                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                                        return function () {
                                                                                                                            return _k13(globalStore, ad.scalar.sub(_result14.actions.length, 1));
                                                                                                                        };
                                                                                                                    }, _address244.concat('_345'), result.value);
                                                                                                                };
                                                                                                            }
                                                                                                        });
                                                                                                    };
                                                                                                }, _address240.concat('_344'), _result15, 3);
                                                                                            };
                                                                                        }, _address240.concat('_343'), _result16);
                                                                                    };
                                                                                }, _address240.concat('_342'), {
                                                                                    model: function (globalStore, _k17, _address243) {
                                                                                        var _currentAddress = _address243;
                                                                                        _addr.save(_globalCurrentAddress, _address243);
                                                                                        return function () {
                                                                                            return sample(globalStore, function (globalStore, _result18) {
                                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                return function () {
                                                                                                    return sum(globalStore, _k17, _address243.concat('_341'), _result18.rewards);
                                                                                                };
                                                                                            }, _address243.concat('_340'), result.value);
                                                                                        };
                                                                                    }
                                                                                });
                                                                            };
                                                                        }, _address240.concat('_339'), function (globalStore, _k19, _address241) {
                                                                            var _currentAddress = _address241;
                                                                            _addr.save(_globalCurrentAddress, _address241);
                                                                            return function () {
                                                                                return Infer(globalStore, _k19, _address241.concat('_338'), {
                                                                                    model: function (globalStore, _k20, _address242) {
                                                                                        var _currentAddress = _address242;
                                                                                        _addr.save(_globalCurrentAddress, _address242);
                                                                                        return function () {
                                                                                            return simulate(globalStore, function (globalStore, s) {
                                                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                                                return function () {
                                                                                                    return _k20(globalStore, s);
                                                                                                };
                                                                                            }, _address242.concat('_337'), policy);
                                                                                        };
                                                                                    },
                                                                                    method: 'forward',
                                                                                    samples: 1000
                                                                                });
                                                                            };
                                                                        });
                                                                    };
                                                                };
                                                                return function () {
                                                                    return enumPolicy(globalStore, function (globalStore, _result1) {
                                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                                        return function () {
                                                                            return run(globalStore, _k0, _address0.concat('_355'), 'Q', _result1);
                                                                        };
                                                                    }, _address0.concat('_354'));
                                                                };
                                                            }, _address0.concat('_336'), {
                                                                vs: _result23,
                                                                ps: _result24
                                                            });
                                                        };
                                                    }, _address0.concat('_335'));
                                                };
                                            }, _address0.concat('_334'), 1, 2);
                                        };
                                    }, _address0.concat('_327'), function (globalStore, _k38, _address231, state, action) {
                                        var _currentAddress = _address231;
                                        _addr.save(_globalCurrentAddress, _address231);
                                        var weights = [
                                            17.78658,
                                            ad.scalar.neg(1.36625),
                                            14.89534,
                                            29.95971
                                        ];
                                        return function () {
                                            return ad.scalar.eq(action, TERM_ACTION) ? termValue(globalStore, _k38, _address231.concat('_322'), state) : VOC_1(globalStore, function (globalStore, _result39) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return VPI_action(globalStore, function (globalStore, _result40) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return VPI_full(globalStore, function (globalStore, _result41) {
                                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                                return function () {
                                                                    return dot(globalStore, _k38, _address231.concat('_326'), weights, [
                                                                        COST,
                                                                        _result39,
                                                                        _result40,
                                                                        _result41
                                                                    ]);
                                                                };
                                                            }, _address231.concat('_325'), state);
                                                        };
                                                    }, _address231.concat('_324'), state, action);
                                                };
                                            }, _address231.concat('_323'), state, action);
                                        };
                                    });
                                };
                            }, _address0.concat('_316'), function (globalStore, _k49, _address227, state) {
                                var _currentAddress = _address227;
                                _addr.save(_globalCurrentAddress, _address227);
                                return function () {
                                    return obsTree(globalStore, function (globalStore, _result52) {
                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                        return function () {
                                            return expectedObservationValue(globalStore, function (globalStore, _result50) {
                                                _addr.save(_globalCurrentAddress, _currentAddress);
                                                return function () {
                                                    return termValue(globalStore, function (globalStore, _result51) {
                                                        _addr.save(_globalCurrentAddress, _currentAddress);
                                                        return function () {
                                                            return _k49(globalStore, ad.scalar.sub(_result50, _result51));
                                                        };
                                                    }, _address227.concat('_315'), state);
                                                };
                                            }, _address227.concat('_314'), _result52);
                                        };
                                    }, _address227.concat('_313'), state, _.range(17));
                                };
                            });
                        };
                    }, _address0.concat('_309'), function (globalStore, _k57, _address225, state) {
                        var _currentAddress = _address225;
                        _addr.save(_globalCurrentAddress, _address225);
                        return function () {
                            return stateTree(globalStore, function (globalStore, _result58) {
                                _addr.save(_globalCurrentAddress, _currentAddress);
                                return function () {
                                    return expectedObservationValue(globalStore, _k57, _address225.concat('_308'), _result58);
                                };
                            }, _address225.concat('_307'), state);
                        };
                    });
                };
            }, _address0.concat('_290'), {
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
    });
});

webppl.runEvaled(main, __runner__, {}, {}, topK, '');