let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
argglobal
if bufexists(fnamemodify("/home/NixOS/coding/takneek26/PS/synapse_fs/serve.cpp", ":p")) | buffer /home/NixOS/coding/takneek26/PS/synapse_fs/serve.cpp | else | edit /home/NixOS/coding/takneek26/PS/synapse_fs/serve.cpp | endif
if &buftype ==# 'terminal'
  silent file /home/NixOS/coding/takneek26/PS/synapse_fs/serve.cpp
endif
setlocal foldmethod=expr
setlocal foldexpr=v:lua.vim.treesitter.foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=99
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
13
sil! normal! zo
40
sil! normal! zo
42
sil! normal! zo
66
sil! normal! zo
let s:l = 11 - ((10 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 11
normal! 029|
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
" vim: set ft=vim :
