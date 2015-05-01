rng('default');
%rng('shuffle');
rng(1);

n = 2000;
p_t = 0.75;

alpha = 6;
beta = 2;

l = 15;
r = 20;

m = n*l/r;


unif = rand(n,1);
% converting uniform distribution to 1 and -1 with probability p_t
tasks = (unif <= p_t)*2 - 1;

% reliablity of workers sampled from beta distribution
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);

% generating a graph - adjacency matrix using configuration model
left_half = reshape(repmat(1:n,l,1),n*l,[]);
right_half = reshape(repmat(1:m,r,1),m*r,[]);

index = randperm(n*l);
right_half = right_half(index);

graph = zeros(n, m);
A = zeros(n, m);

for i = 1:(n*l)
  graph(left_half(i), right_half(i)) = 1;
  A(left_half(i), right_half(i)) = -tasks(left_half(i));
  if (rand() <= p(right_half(i)))
    A(left_half(i), right_half(i)) = tasks(left_half(i));
  end
end

q = zeros(n, 2);
% initialisation
for i = 1:n
  q(i, 1) = sum(A(i, :) == -1)/sum(graph(i, :) == 1);
  q(i, 2) = sum(A(i, :) == 1)/sum(graph(i, :) == 1);
end

display('just majority voting')
[ row_max row_argmax ] = max( q, [], 2 );
c = (row_argmax*2 - 3);
successful = sum(tasks == c)
total = n

pj = zeros(1, m);
for j = 1:m
  dj = find(graph(:, j) == 1);
  pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
end

for k = 1:100
  % E-step
  p_vals = [1-pj;pj]';
  for i = 1:n
    di = find(graph(i, :) == 1);
    ti = 1;
    prod_plus = prod(p_vals((A(i, di) == ti)*m + di));
    ti = -1;
    prod_minus = prod(p_vals((A(i, di) == ti)*m + di));
    q(i, 1) = prod_minus / (prod_minus + prod_plus);
    q(i, 2) = prod_plus / (prod_minus + prod_plus);
  end

  % M-step
  for j = 1:m
    dj = find(graph(:, j) == 1);
    pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
  end
end

display('running done')
[ row_max row_argmax ] = max( q, [], 2 );
c = (row_argmax*2 - 3);
successful = sum(tasks == c)
total = n