function result = sigmoid(beta, theta)
    result = 1./(1+exp(-(beta-theta)));
end