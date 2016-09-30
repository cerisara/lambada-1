
-- This file trains and tests the RNN from a batch loader.
require('nn')
require('nnx')
require('nngraph')
require('options')
require 'utils.misc'

require('utils.batchloader')
require('utils.textsource')
require('models.meta')

-- Parse arguments
local cmd = RNNOption()
g_params = cmd:parse(arg)
torch.manualSeed(1990)

-- cuda?
local cuda = false
if g_params.cuda_device then
    require 'cutorch'
    require 'cunn'
    require 'cunnx'
    cutorch.setDevice(g_params.cuda_device)
    cutorch.manualSeed(1990)
    cuda = true
end

-- build the torch dataset
local g_dataset = TextSource(g_params.dataset)
local vocab_size = g_dataset:get_vocab_size()
local use_lambada = false
if g_params.dataset.name == 'lambada' then
    use_lambada = true
end

-- Create clusters if hierarchical softmax is used
if string.find(g_params.model.name, '_hsm') then
    g_dataset:create_clusters(g_params.dataset)
elseif string.find(g_params.model.name, "_smt") then
    g_dataset:create_frequency_tree(g_params.dataset)
    use_lambada = false -- smt can't be used to test lambada
end

if string.find(g_params.model.name, 'scrnn_') then
    g_params.model.context_scale = 0.05
    g_params.model.n_layers = 1
end

local g_dictionary = g_dataset.dict
-- A data sampler for training and testing
batch_loader = BatchLoader(g_params.dataset, g_dataset)


local function eval(split_idx)

    split_idx = split_idx or 2
	local inputs, targets = batch_loader:next_batch(split_idx)

	if cuda == true then
		inputs = inputs:float():cuda()
		targets = targets:float():cuda()
	end

	local loss = meta:eval(inputs, targets)

	return loss / math.log(2)
end



-- Training function
-- The batch loader keeps sampling tensors from the data
-- Tensors are fed into the meta trainer
-- Output average loss and total words 
local function train_epoch(learning_rate)

    local ntrain = batch_loader.ntrain
    local total_length = 0
    local total_loss = 0
    local total_words = 0
    meta:reset() -- reset the initial state (to zeros)
    batch_loader:reset_batch_pointer()

    for i = 1, ntrain do
        xlua.progress(i, ntrain)
        local inputs, targets = batch_loader:next_batch(split_idx)

        if cuda == true then
            inputs = inputs:float():cuda()
            targets = targets:float():cuda()
        end

        local loss, length, batch_size = meta:train(inputs, targets, learning_rate)

        total_loss = total_loss + loss
        total_length = total_length + length
        -- Count number of words
        total_words = total_words + length * batch_size

        if i % 50 == 0 then
            collectgarbage()
        end

        if sys.isNaN(loss) then
            print('Warning ... Not a Number detected')
            os.exit(0) 
        end
    end

    local train_loss = total_loss / total_length / math.log(2)

    return train_loss, total_words

end


local function evaluate_lambada(cuda, topn)

    local lambada_streams = batch_loader:get_lambada_streams()
    local n_samples = #lambada_streams
    local total_n = 0
    local total_acc = 0
    local total_err = 0
    local total_unk = 0
    
    -- Update hsm in cpu if use hsm
    meta:update_cpu_hsm()
    
    if meta.cpu_hsm then
        meta.cpu_hsm:change_bias()
    end

    -- Process each of them
    for k, stream in pairs(lambada_streams) do
      
      xlua.progress(k, n_samples)
          
      total_n = total_n + 1

      local inputs = stream[{{1, stream:size(1)-1}}]
      local label = stream[-1]

      if cuda == true then
        inputs = inputs:cuda()
      end

			print(label)
      local err, acc = meta:lambada(inputs, label, topn)
      if label[1] == 1 then -- unknown word
        acc = 0 -- automatically fail for unknown word
        total_unk = total_unk + 1
      end
      total_err = total_err + err
      total_acc = total_acc + acc

      collectgarbage()

    end

    local loss = total_err / total_n / math.log(2)

    local accuracy = total_acc / total_n

    print(string.format('Lambada Validation: Entropy (base 2) : %.5f || ' ..
                                 'Perplexity : %0.5f || ' .. 'Accuracy : %0.3f with total ' .. ' %i unknown words',
                             loss, math.pow(2, loss), accuracy * 100, total_unk))

    return loss, accuracy

end

local function run(config, model_config, dictionary, lambada, cuda)

    lambada = lambada or false
    -- create the directory to save trained models
    if config.save_dir ~= nil then
       if paths.dirp(config.save_dir) == false then
           os.execute('mkdir -p ' .. config.save_dir)
       end
       print('*** models will be saved after each epoch ***')
    end

    local last_model = nil
    local train_err = {}
    local val_err = {}
    local patience = 0
    local learning_rate = config.initial_learning_rate
    local learning_rate_shrink =config.learning_rate_shrink
    local shrink_type = config.shrink_type
    local lambada_loss, lambada_acc, val_loss
    local load_state

    -- Load trained models 

    if config.load ~= '' then

        load_state = torch.load(config.load)
        -- meta.protos = save_state.protos
        -- learning_rate = save_state.learning_rate
        learning_rate_shrink = load_state.learning_rate_shrink
        print("Model parameters loaded from " .. config.load)
        -- print("Learning rate loaded to " .. learning_rate)
    
    end

    meta = MetaRNN(g_params.model, g_dictionary, cuda, load_state)


    val_loss = eval(2) 
    val_err[0] = val_loss

    if lambada == true then 
        lambada_loss, lambada_acc = evaluate_lambada(cuda, config.topn) 
        val_err[0] = lambada_loss
    end

   

    print(string.format('\nValidation: Entropy (base 2) : %.5f || ' ..
                             'Perplexity : %0.5f',
                        val_loss, math.pow(2, val_loss)))


    for epoch = 1, config.n_epochs do

        -- Stop after a number of epochs without progress
        if patience >= 3 then break end

        local timer = torch.tic()

        -- Train one epoch here
        local train_loss, total_words = train_epoch(learning_rate)

        local elapse = torch.toc(timer)

        print(string.format('\n\nEpoch: %d. Training time: %.2fs. ' ..
                                 'Words/s: %.2f',
                             epoch, elapse, total_words/elapse))
        print(string.format('\nTraining: Entropy (base 2) : %.5f || ' ..
                                 'Perplexity : %0.5f',
                             train_loss, math.pow(2, train_loss)))

        train_err[epoch] = train_loss

        val_loss = eval(2)

        val_err[epoch] = val_loss

        if lambada == true then 
            
            local lambada_loss, lambada_acc = evaluate_lambada(cuda, config.topn)
            val_err[epoch] = lambada_loss
        end

        print(string.format('\nValidation: Entropy (base 2) : %.5f || ' ..
                             'Perplexity : %0.5f',
                         val_loss, math.pow(2, val_loss)))

       
        -- Decrease lr if loss increase
        if val_err[epoch] > val_err[epoch-1] * config.shrink_factor then

            if last_model ~= nil then
                meta.protos = last_model
            end 

            learning_rate = learning_rate / config.learning_rate_shrink
            print('\nDecreasing the learning rate to ' .. learning_rate)
            patience = patience + 1
        else
            last_model = meta.protos
            patience = 0
        end

        -- Save models if opted
        if config.save_dir ~= nil then
            local save_state = {}
            save_state.protos = meta.protos
            save_state.learning_rate = learning_rate
            save_state.learning_rate_shrink = config.learning_rate_shrink
            torch.save(paths.concat(config.save_dir, 'model_' .. epoch), save_state)
        end


    end

    -- Get test perplexity
    local test_err = eval(3)

    print(string.format('\nTesting: Entropy (base 2) : %.5f || '..
                               'Perplexity : %0.5f',
                           test_err, math.pow(2, test_err)))





end
-- print(g_dictionary)
cmd:print_params(g_params)

run(g_params.trainer, g_params.model, g_dictionary, use_lambada, cuda)


