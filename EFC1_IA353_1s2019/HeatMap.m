function status = HeatMap(W)
  [nl,nc] = size(W);
% [nl,nc] = size(Xt);
  for ind = 1:nc,
      k = 1;
      for i=1:28,
          for j=28:-1:1,
              v(i,j) = W(k, ind);
  %             v(i,j) = Xt(ind,k);
              k = k+1;
          end
      end
      figure(ind)
      pcolor([1:28],[1:28],v');
      colorbar;

  end
end