function acertos = calc_perc(mtxDados, mtxVerdade)
  total = length(mtxVerdade);
  total_acertos = 0;
  if (total != length(mtxDados))
    acertos = mtxDados'.*mtxVerdade;
  else
    for i = 1:1:length(mtxVerdade)
      [~, ind] = max(mtxDados(i,:));
      if (mtxVerdade(i, ind) == 1)
        total_acertos += 1; 
      endif
    endfor
  endif
  
  acertos = total_acertos/total;
endfunction