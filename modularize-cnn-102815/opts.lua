-- opts.lua for parsing the input parameters

local optsParser = {}

function optsParser.parse(arg)

	local cmd = torch.CmdLine()
	cmd:text() -- log a custom text message
	cmd:text('Training a simple CNN')
	cmd:text()
	cmd:text('Options')
	cmd:option('-lr',0.01,'Learning rate')
	cmd:option('-me',10,'Maximum Epochs')
	cmd:option('-bs',50,'Batch size')
	cmd:option('-optim','adagrad','Optimization method')

	cmd:text()

	-- parse the input params
	params =cmd:parse(arg)

	return params
end

return optsParser






