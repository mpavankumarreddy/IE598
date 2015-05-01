rng('default');
%rng('shuffle');
rng(1);

n = 30;
p_t = 0.75;

alpha = 6;
beta = 2;

l = 2;
r = 3;

m = n*l/r;


unif = rand(n,1);
% converting uniform distribution to 1 and -1 with probability p_t
tasks = (unif <= p_t)*2 - 1;

% reliablity of workers sampled from beta distribution
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);

% generating a graph - adjacency matrix using configuration model
left_half = reshape(repmat(1:n,l,1),n*1,[]);
right_half = reshape(repmat(1:m,r,1),m*r,[]);

index = randperm(n*l);
right_half = right_half(index);

graph = zeros(n, m);
A = zeros(n, m);

for i = 1:(n*l)
  graph(left_half(i), right_half(i)) = 1;
  A(left_half(i), right_half(i)) = (rand() <= p(right_half(i)))*2 - 1;
end

domain_p = linspace(0.1, 0.99, 20);
dist_p = betapdf(domain_p, alpha, beta);

v_t_w = zeros(n,m*2);
v_w_t = zeros(m, n*numel(domain_p));

% v_w_t is in the shape of 2 X m*n. Its like the first
% m columns correspond to the messages sent from first task to all m
% tasks
%v_t_w = reshape(reshape(repmat(graph', 2, 1), m, n*2)', 2, []);
%repmat(reshape(a', 1, []),2,1)
v_t_w = reshape(reshape(repmat(graph(:,:)', 2, 1), m, [])', 2, []);

% v_w_t is in the shape of numel(domain_p) X m*n. Its like the first
% n columns correspond to the messages sent from first worker to all n
% tasks
v_w_t = reshape(reshape(repmat(graph', numel(domain_p), 1), m, n*numel(domain_p))', numel(domain_p), []);