  // Complementary error function
  // From Numerical Recipes in C 2e p221
  var erfc = function(x) {
    var z = Math.abs(x);
    var t = 1 / (1 + z / 2);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 +
            t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
            t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))))
    return x >= 0 ? r : 2 - r;
  };

  // Inverse complementary error function
  // From Numerical Recipes 3e p265
  var ierfc = function(x) {
    if (x >= 2) { return -100; }
    if (x <= 0) { return 100; }

    var xx = (x < 1) ? x : 2 - x;
    var t = Math.sqrt(-2 * Math.log(xx / 2));

    var r = -0.70711 * ((2.30753 + t * 0.27061) /
            (1 + t * (0.99229 + t * 0.04481)) - t);

    for (var j = 0; j < 2; j++) {
      var err = erfc(r) - xx;
      r += err / (1.12837916709551257 * Math.exp(-(r * r)) - r * err);
    }

    return (x < 1) ? r : -r;
  };

  // Probability density function
function normPDF(x,mu,sigma) {
    var m = sigma * Math.sqrt(2 * Math.PI);
    var e = Math.exp(-Math.pow(x - mu, 2) / (2 * Math.pow(sigma,2)));
    return e / m;
  };

  // Cumulative density function
  function normCDF(x,mu,sigma){
    return 0.5 * erfc(-(x - mu) / (sigma * Math.sqrt(2)));
  };

  // Percent point function
  function percentile(x,mu,sigma){
    return mu - sigma * Math.sqrt(2) * ierfc(2 * x);
  };


function maxOfGaussiansPDF(x,mu,sigma){
    //returns the probability density p(max {X_0,...,X_n}) where X_i ~ N(mu[i],sigma[i]).
    //This function should always return approximately the same value as MatlabTools/pdfOfMaxOfGaussians.m
    //This functions expects that all elements of sigma are greater than zero and throws an error otherwise.
    
    if (sigma.some(function(x){return x<=0})){
        throw new Error('maxOfGaussiansPDF: All sigmas have to be greater than 0.','gaussian.js',56)
    }
    
    if (typeof(x)=="number"){
		x=[x]
	}

    var n=mu.length

    var cumulative_distributions= new Array()
    var densities= new Array()
    for (var k=0; k<n; k++){
        density = new Array();
        cumulative_distribution = new Array()
        
        for (var i=0; i<x.length; i++){
            density.push(normPDF((x[i]-mu[k])/sigma[k], 0, 1))
            cumulative_distribution.push(normCDF((x[i]-mu[k])/sigma[k],0,1))
        }
        densities.push(density)
        cumulative_distributions.push(cumulative_distribution)
    }

    var p = new Array(x.length);
    
    for (var i=0; i<x.length; i++){
        p[i]=0;
        for (var l=0; l<n; l++){
            others=_.without(_.range(n),l);        
            
            var cdf_product=1;
            for (var k=0; k<others.length; k++){
                cdf_product*=cumulative_distributions[others[k]][i]
            }
            
            p[i]=p[i]+1/sigma[l]*densities[l][i]*cdf_product;
        }
    }
    
    return p
    
}

function EVOfMaxOfGaussians(mu,sigma){
//[E_max,STD_max]=EVofMaxOfGaussians(mu,sigma) computes 
//E[max {X_1,...,X_n}] and STD[max {X_1,...,X_n}] for n normally distributed
//random variables X_1,...,X_n with means mu(1),...,mu(n), and standard
//deviations sigma(1),...,sigma(n).
//The output of this function should be equal to that of 
//MatlabTools/EVofMaxOfGaussians.m.

var max_sigma=getMaxOfArray(sigma)    
    
if (mu.length==1){
    return [mu[0],sigma[0]]
}
if (sigma.every(function(x) {return x==0})){
    E_max=getMaxOfArray(mu);
    STD_max=0;
}
else{
    if (sigma.every(function(x) {return x>0})){
        x_min=getMinOfArray(mu)-4*max_sigma;
        x_max=getMaxOfArray(mu)+4*max_sigma;
        E_max=integral(x_min,x_max, 0.01*max_sigma/2,  function(x) {return x*maxOfGaussiansPDF(x,mu,sigma)});
        STD_max=Math.sqrt( integral(x_min,x_max, 0.01*max_sigma/2, function(x) {return Math.pow(x-E_max,2)*maxOfGaussiansPDF(x,mu,sigma)}));
    }
    else{   //some outcomes are known and some are unknown 
        for_sure= new Array()
        mu_uncertain= new Array() 
        sigma_uncertain= new Array() 
        for (i=0; i<sigma.length; i++){
            if (sigma[i]==0){
                for_sure.push(mu[i])
            }
            else{
                mu_uncertain.push(mu[i])
                sigma_uncertain.push(sigma[i])
            }
        }
        max_for_sure=getMaxOfArray(for_sure)
    
        x_min=getMinOfArray(mu)-4*max_sigma;
        x_max=getMaxOfArray(mu)+4*max_sigma;
    
        E_max=integral(x_min,x_max, 0.01*max_sigma/2, function(x){return Math.max(max_for_sure,x)*maxOfGaussiansPDF(x,mu_uncertain,sigma_uncertain)});
        STD_max=Math.sqrt( integral(x_min,x_max, 0.01*max_sigma/2,  function(x){return Math.pow(Math.max(max_for_sure,x)-E_max,2) *maxOfGaussiansPDF(x,mu_uncertain,sigma_uncertain)}));
    }
}
    return [E_max,STD_max]

}


function integral(a, b, dx, f) {
//This function returns the Euler's rule approximation to the integral of f from a to b with step size dx.
	
	// calculate the number of trapezoids
	var n = (b - a) / dx;
	
	// define the variable for area
	var Area = 0;
	
	//loop to calculate the area of each trapezoid and sum.
	for (var i = 1; i <= n; i++) {
		//the x locations of the left and right side of each trapezpoid
	    var	x0 = a + (i-1)*dx;
		var x1 = a + i*dx;
		
		// the area of each trapezoid
		var Ai = dx * (f(x0) + f(x1))/ 2.;
		
		// cumulatively sum the areas
		Area = Area + Ai	
        
        //console.log("Loop iteration "+i)
		
	} 
	return Area;
}

function ETruncatedNormal(mu,sigma,x_min,x_max){
    //ETruncatedNormal(mu,sigma,x_min,x_max) returns the value of $\int_a^b normpdf(x,mu)*min(max(x,x_min),x_max)    dx$
    
    realmax=1000000;
    
    int_below=normCDF(x_min,mu,sigma)*_.max([-realmax,x_min]);
    int_above=(1-normCDF(x_max,mu,sigma))*_.min([realmax,x_max]);

    int_middle=mu*(normCDF(x_max,mu,sigma)-normCDF(x_min,mu,sigma))-sigma**2*(normPDF(x_max,mu,sigma)-normPDF(x_min,mu,sigma));
    
    EV=int_below+int_middle+int_above;
    
    return EV
    
}
                     
                     
function getMaxOfArray(numArray) {
  //returns the largest value in numArray even if the array and ignores NaN and null entries.
  return _.max(numArray)//Math.max.apply(null, numArray);
}
        
function getMinOfArray(numArray) {
  //returns the smallest value in numArray even if the array and ignores NaN and null entries.    
  return _.min(numArray)//Math.min.apply(null, numArray);
}