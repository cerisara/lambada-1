
-- This file implements an intermediate class between the textsource and the trainer
-- The textsource outputs a text stream (with batch size)
-- And the batch loader will select the batch (with seq length) and give the trainer

local BatchLoader = torch.class('BatchLoader')

function BatchLoader:__init(config, text_source)

	self.text_source = text_source
	self.seq_length = config.seq_length
	self.batch_size = config.batch_size
	self.vocab_size = self.text_source:get_vocab_size()

	self.split_sizes = {} 
	self.all_batches = {} 
	self.length = {}

	local streams = {}

	streams[1] = text_source:get_stream('train')
	streams[2] = text_source:get_stream('valid')
	streams[3] = text_source:get_stream('test')

	for i = 1, 3 do

		local stream = streams[i]
		-- Convert the stream to inputs and labels
		local inputs = stream[{{1, stream:size(1)-1}}]
    	local labels = stream[{{2, stream:size(1)}}]

    	-- Store in the table
    	self.all_batches[i] = {inputs, labels}
    	local nbatches = math.ceil(inputs:size(1) / self.seq_length) -- dim 1 is the number of elements, dim 2 is batch size
    	self.split_sizes[i] = nbatches -- number of batches
    	if i > 1 then
    		self.split_sizes[i] = 1 -- one batch for valid and test (we don't need to backward in testing)
    	end
	end

	self.batch_idx = {0, 0, 0}
	self.ntrain = self.split_sizes[1]

	print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', 
          self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()

end

function BatchLoader:get_vocab_size()
	return self.vocab_size
end

function BatchLoader:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoader:next_batch(split_idx)

	split_idx = split_idx or 1
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
     

   	if split_idx <= 1 then
   		local start_id = self.seq_length * (idx - 1) + 1
    	local max_length = self.all_batches[split_idx][1]:size(1)
    	local end_id = math.min(self.seq_length * (idx), max_length) 
	    local x = self.all_batches[split_idx][1]:sub(start_id, end_id, 1, -1)
	    local y = self.all_batches[split_idx][2]:sub(start_id, end_id, 1, -1)
	    	
	    -- x, y should have dim (seq_length * batch_size)
	    return x, y

	-- One stream for test set and valid
	else
		return self.all_batches[split_idx][1], self.all_batches[split_idx][2]
    end
end

function BatchLoader:get_lambada_streams()

	return self.text_source:get_lambada_streams()
end
