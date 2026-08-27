let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
argglobal
if bufexists(fnamemodify("/home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp", ":p")) | buffer /home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp | else | edit /home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp | endif
if &buftype ==# 'terminal'
  silent file /home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp
endif
setlocal foldmethod=expr
setlocal foldexpr=v:lua.vim.treesitter.foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=99
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
25
sil! normal! zo
41
sil! normal! zo
let s:l = 43 - ((23 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 43
normal! 013|
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
" vim: set ft=vim :
