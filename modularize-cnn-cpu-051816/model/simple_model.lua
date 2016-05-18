-- model file

-- Make sure no variable declaration outside function. It will be confused

require 'nn'

function simple_model(nb_outputs)
	local cnn = nn.Sequential();

	-- first convolution, non-linear, and pooling
	cnn:add(nn.SpatialConvolution(1,100,3,3,1,1,0,0)) -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]). E.g., input:1x16x16, beomes 10x14x14
	cnn:add(nn.ReLU()) -- non-linear layer
	cnn:add(nn.SpatialMaxPooling(2,2)) -- becomes 10x7x7

	-- second convolution, non-linear, and pooling
	cnn:add(nn.SpatialConvolution(100,400,2,2,1,1,0,0)) -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]). E.g., input:10x14x14, beomes 40x6x6
	cnn:add(nn.ReLU()) -- non-linear layer
	cnn:add(nn.SpatialMaxPooling(2,2)) -- becomes 40x3x3

	cnn:add(nn.Reshape(400*3*3))
	cnn:add(nn.Linear(400*3*3,nb_outputs))
	cnn:add(nn.LogSoftMax())

	return cnn:float()
end




