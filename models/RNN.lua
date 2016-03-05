local RNN = {}

-- local ok, cunn = pcall(require, 'fbcunn')

function RNN.rnn(input_size, rnn_size,  n, dropout, hsm, nonlinear)
  
  nonlinear = nonlinear or "sigmoid"
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]

  end
  local embedding_layer = nn.LookupTable(input_size, rnn_size)

  local nonlinear_function

  if nonlinear == "sigmoid" then
    nonlinear_function = nn.Sigmoid()
  elseif nonlinear == "tanh" then
    nonlinear_function = nn.Tanh()
  else
    nonlinear_function = nn.ReLU()
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    
    local prev_h = inputs[L+1]
    if L == 1 then 
      -- x = OneHot(input_size)(inputs[1])
      x = embedding_layer(inputs[1])
      input_size_L = rnn_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h):annotate{name="h2h_" .. L}
    local next_h = nonlinear_function(nn.CAddTable(){i2h, h2h})
    -- local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})


    table.insert(outputs, next_h)
  end

  local top_h = outputs[#outputs]
  if dropout > 0 then 
        top_h = nn.Dropout(dropout)(top_h) 
  else
      top_h = nn.Identity()(top_h) --to be compatiable with dropout=0 and hsm>1
  end
-- set up the decoder
  if hsm == false then
    local proj = nn.Linear(rnn_size, input_size)(top_h)
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
  else
    -- if HSM is used then softmax will be done later
    table.insert(outputs, top_h)

  end



  return nn.gModule(inputs, outputs)
end

return RNN
