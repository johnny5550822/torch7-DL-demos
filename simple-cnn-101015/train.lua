-- ###############
-- This is a minimal implementation (e.g., no batch processing) of a CNN on digit (0-9) classification. This code is largely obtained from luacnn, https://github.com/hpenedones/luacnn
-- The training data can be also obtained in https://github.com/hpenedones/luacnn

-- ###############

require "torch"
require "nn"
require "math"

-- variables
learningRate = 0.01
maxIterations = 100000

-- create the neural network
function create_network(nb_outputs)
	local cnn = nn.Sequential();	

    -- filter configuration
    nInputPlane = 1; -- number of inputPlane
    nOutputPlane = 10; --number of maps
    filterSize = 5;
    stride = 1;
    padding = 0;

    -- pooling configuration
    

	cnn:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,filterSize,fitlerSize,stride,padding))











































