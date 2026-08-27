let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
argglobal
if bufexists(fnamemodify("/home/NixOS/coding/takneek26/PS/synapse_fs/spp", ":p")) | buffer /home/NixOS/coding/takneek26/PS/synapse_fs/spp | else | edit /home/NixOS/coding/takneek26/PS/synapse_fs/spp | endif
if &buftype ==# 'terminal'
  silent file /home/NixOS/coding/takneek26/PS/synapse_fs/spp
endif
setlocal foldmethod=expr
setlocal foldexpr=v:lua.vim.treesitter.foldexpr()
setlocal foldmarker={{{,}}}
setlocal foldignore=#
setlocal foldlevel=99
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldenable
let s:l = 1 - ((0 * winheight(0) + 14) / 29)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 011|
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
" vim: set ft=vim :
