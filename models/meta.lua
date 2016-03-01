
local MetaRNN = torch.class('MetaRNN')
local model_utils = require 'utils.model_utils'
local LSTM = require 'models.LSTM'
local RNN = require 'models.RNN'
require('models.HLogSoftMax')


-- A wrapping class than handles RNN computation
function MetaRNN:__init(config, dict, cuda)

	self.batch_loader = batch_loader
	self.config = config
	self.cuda = false

	local vocab_size = #dict.index_to_symbol
	print('creating an ' .. config.name .. ' with ' .. config.n_layers .. ' layers')
	-- create RNN models here
	self.protos = {}


	self.hsm = false
    self.smt = false
    local houtput = true
	if string.find(config.name, 'hsm') then
		self.hsm = true -- Use a hierarchical softmax (HSM or SMT). 
		-- This would tell the model to not decode at the highest layer
	elseif string.find(config.name, "_smt") then
        self.smt = true
    else
        houtput = false
    end

	-- create the rnn steps
	if string.find(config.name, 'lstm_') then
    	self.protos.rnn = LSTM.lstm(vocab_size, config.n_hidden, config.n_layers, config.dropout, houtput)
    elseif string.find(config.name, 'srnn_') then
    	self.protos.rnn = RNN.rnn(vocab_size, config.n_hidden, config.n_layers, config.dropout, houtput)
    end

    local encoded_size = config.n_hidden -- Size of the last hidden layer (encoded input)
    if self.hsm == true then 
        self.protos.criterion = nn.HLogSoftMax(dict.clusters, encoded_size)
    elseif self.smt == true then
        self.protos.criterion = nn.TreeNLLCriterion()
        self.protos.softmaxtree = nn.SoftMaxTree(encoded_size, dict.tree, dict.root_id, false, false)
    else
    	self.protos.criterion = nn.ClassNLLCriterion()
    end

    

    if cuda == true then
    	self:transfer_gpu()
    else
        self.type = 'torch.DoubleTensor'
    end



    self.params, self.grad_params = model_utils.combine_all_parameters(self.protos.rnn)
    if self.hsm == true then
    	self.hsm_params, self.hsm_grad_params = self.protos.criterion:getParameters()
    elseif self.smt == true then
        self.smt_params, self.smt_grad_params = model_utils.combine_all_parameters(self.protos.softmaxtree)
    end

    self:init_params()
    self:unroll(self.config.backprop_len)
    self:reset()


end

function MetaRNN:reset()

    self.train_state = self:make_init_state(self.config.batch_size)

end

function MetaRNN:transfer_gpu()

	self.cuda = true
	self.type = 'torch.CudaTensor'
	for k, v in pairs(self.protos) do v:cuda() end
end

function MetaRNN:make_init_state(batch_size)
    
    local state = {}
    for L=1, self.config.n_layers do
     
		 local h_init = torch.zeros(batch_size, self.config.n_hidden):type(self.type)
		 -- if self.cuda == true then h_init = h_init:cuda() end
		 table.insert(state, h_init:clone())
		 if string.find(self.config.name, 'lstm') then
		     -- since lstm has two memory nodes
		     table.insert(state, h_init:clone())
		 end
    end

    return state

end

function MetaRNN:init_params()

	self.grad_params:zero()
	self.params:uniform(-self.config.initial_val, self.config.initial_val)

	if self.hsm == true then
		self.hsm_params:uniform(-self.config.initial_val, self.config.initial_val)
		self.hsm_grad_params:zero()
    elseif self.smt == true then
        self.smt_params:uniform(-self.config.initial_val, self.config.initial_val)
        self.smt_grad_params:zero()
	end
end

function MetaRNN:unroll(num_steps)

	self.clones = {}
	for name,proto in pairs(self.protos) do
	    print('cloning ' .. name)
	    self.clones[name] = model_utils.clone_many_times(proto, num_steps, not proto.parameters)
	end
end

-- Forward and Backward in one function
-- Inputs are 2D tensors of (bz * seq_length), same as targets
-- Outputs are loss and number of steps for this batch segment
function MetaRNN:train(inputs, targets, learning_rate)
	
    local rnn_states = {[0] = self.train_state}
    local predictions = {}           -- softmax outputs
    local loss = 0
    local batch_size = inputs:size(2)

    self.grad_params:zero()
    if self.hsm_grad_params then self.hsm_grad_params:zero() end

    -- Length of inputs (sometimes it is not equal to seq_length)
    local truncate = inputs:size(1)
    local smt_output = {}
    -- FORWARD PASS 
    for t=1,truncate do
        self.clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)

        -- forward inputs: x and H
        local lst = self.clones.rnn[t]:forward{inputs[t], unpack(rnn_states[t-1])}
        rnn_states[t] = {}
        for i=1,#self.train_state do table.insert(rnn_states[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        
        local err
        -- get loss from criterion
        if self.smt == true then
            smt_output[t] = self.clones.softmaxtree[t]:forward{predictions[t], targets[t]}
            err = self.clones.criterion[t]:forward(smt_output[t])
        else
            err = self.clones.criterion[t]:forward(predictions[t], targets[t]) 
        end

        loss = loss + err
    end

    self.train_state = clone_list(rnn_states[truncate])
   
    
    -- BACKWARD PASS
    local drnn_states = {[truncate] = self:make_init_state(batch_size)} 
    for t=truncate,1,-1 do
        -- backprop through loss, and softmax/linear
        -- compute dL/do for the rnn block
        
        local doutput_t, dsmt_t
        if self.smt == false then
            doutput_t = self.clones.criterion[t]:backward(predictions[t], targets[t]) 
        else
            dsmt_t = self.clones.criterion[t]:backward(smt_output[t], targets[t])
            doutput_t = self.clones.softmaxtree[t]:backward({predictions[t], targets[t]}, dsmt_t)[1]
        end

        table.insert(drnn_states[t], doutput_t)
        
        -- compute dL / dI for rnn block
        local dlst = self.clones.rnn[t]:backward({inputs[t], unpack(rnn_states[t-1])}, drnn_states[t]) 
        

        drnn_states[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then 
                -- k > 1 because we don't need dL / dx
                -- dL / dH: input of this state is output of last state
                drnn_states[t-1][k-1] = v
            end
        end
    end

    self:updateParameters(learning_rate)

    return loss, truncate, batch_size
end

-- Update Params of the RNN 
function MetaRNN:updateParameters(learning_rate)

    -- First clip the gradients
    local grad_norm
    
    -- Compute the gradient norms
    if self.hsm == true then
        grad_norm = torch.sqrt(self.grad_params:norm()^2 + self.hsm_grad_params:norm()^2)
    elseif self.smt == true then
        -- print(self.smt_grad_params:norm())
        grad_norm = torch.sqrt(self.grad_params:norm()^2 + self.smt_grad_params:norm()^2)
    else
        grad_norm = self.grad_params:norm()
    end

    if grad_norm > self.config.gradient_clip then
        local shrink_factor = self.config.gradient_clip / grad_norm
        -- Here we clip the gradients (not sure while clamp doesn't work !!!)
        self.grad_params:mul(shrink_factor)
        if self.hsm == true then
            self.hsm_grad_params:mul(shrink_factor)
        elseif self.smt == true then
            self.smt_grad_params:mul(shrink_factor)
        end
    end

    -- Simple SGD (may add something fancier in the future)
	self.params:add(-learning_rate, self.grad_params)

    if self.hsm == true then
        self.hsm_params:add(-learning_rate, self.hsm_grad_params)
    elseif self.smt == true then
        self.smt_params:add(-learning_rate, self.smt_grad_params)
    end
end

-- Compute loss for a series of inputs and targets
-- Both inputs are 2D tensors (length * batch)
-- Can input any batch 
function MetaRNN:eval(inputs, targets)

	local batch_size = inputs:size(2) -- dim n_samples * n_batch_size
	if self.hsm == true then
        self.protos.criterion:change_bias()
    end
    self.protos.rnn:evaluate()

    -- local test_state = 
    local rnn_state = self:make_init_state(batch_size)
    local n_layers = #rnn_state
    local length = inputs:size(1)
    local loss = 0
    local prediction

    for t = 1, length do
        -- xlua.progress(t, length)
        local lst = self.protos.rnn:forward{inputs[t], unpack(rnn_state)}
        rnn_state = {}
        for i=1, n_layers do table.insert(rnn_state, lst[i]) end
        prediction = lst[#lst] 
        
        local loss_per_step
        local smt_output
        if self.smt == false then
            loss_per_step = self.protos.criterion:forward(prediction, targets[t])
        else
            smt_output = self.protos.softmaxtree:forward{prediction, targets[t]}
            loss_per_step = self.protos.criterion:forward(smt_output, targets[t])
        end
        loss = loss + loss_per_step
    end
    loss = loss / length

    return loss
end


-- Evaluate the lambada loss (ppl and/or accuracy)
function MetaRNN:lambada(inputs, target, topn)

    topn = topn or 100
    local batch_size = inputs:size(2)
    assert(batch_size == 1) -- Batch size must be 1 

    if self.hsm == true then
        self.protos.criterion:change_bias()
    end

    local rnn_state = self:make_init_state(batch_size)
    local n_layers = #rnn_state
    local length = inputs:size(1)
    local top_layer

    for t = 1, length do 

        local lst = self.protos.rnn:forward{inputs[t], unpack(rnn_state)}
        rnn_state = {}
        for i =1, n_layers do table.insert(rnn_state, lst[i]) end

        if t == length then
            top_layer = lst[#lst] 
        end
    end

    -- Compute perplexity ( log P(y|x))
    local loss

    -- Compute accuracy (argmax P(y|x) with topn)
    local acc = 0
    local sorted_value, sorted_indx

    if self.hsm == true then    
        -- Swap back to CPU for faster testing 
        local criterion = self.protos.criterion:clone()
        if self.type == 'torch.CudaTensor' then
            criterion:float()
            top_layer = top_layer:float()
        end

        local prob_dist, prob = criterion:generateDistribution(top_layer, target)

        loss = prob

        sorted_value, sorted_indx = torch.sort(prob_dist, false)

        for i = 1, topn do
            -- print(target[1], sorted_indx[i], prob_dist[sorted_indx[i]])
            if target[1] == sorted_indx[i] then
                acc = 1
                break
            end
        end

    else -- Normal softmax (much faster than HSM)
        loss  = self.protos.criterion:forward(top_layer, target)
        -- The distribution of all words in vocab
        local prob_dist = top_layer
        sorted_value, sorted_indx = torch.sort(prob_dist, 2, false)

        for i = 1, topn do
            if target[1] == sorted_indx[1][i] then
                acc = 1
                break
            end
        end
    
    end

    return loss, acc 


end
