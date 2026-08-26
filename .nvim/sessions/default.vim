let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +11 /home/NixOS/coding/takneek26/PS/synapse_fs/spp.cpp
badd +46 term:///home/NixOS/coding/takneek26/PS/synapse_fs//1766708:/run/current-system/sw/bin/bash
badd +61 term:///home/NixOS/coding/ctf/pclub//1767643:/run/current-system/sw/bin/bash
badd +13 term:///home/NixOS/coding/takneek26/PS/synapse_fs//1767701:/run/current-system/sw/bin/bash
badd +0 term:///home/NixOS/coding/takneek26/PS/synapse_fs//1795284:/run/current-system/sw/bin/bash
argglobal
%argdel
edit /home/NixOS/coding/takneek26/PS/synapse_fs/spp.cpp
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 150 + 120) / 240)
exe '2resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 2resize ' . ((&columns * 89 + 120) / 240)
exe '3resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 3resize ' . ((&columns * 89 + 120) / 240)
argglobal
let s:l = 10 - ((9 * winheight(0) + 30) / 61)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 10
normal! 0
wincmd w
argglobal
if bufexists(fnamemodify("term:///home/NixOS/coding/takneek26/PS/synapse_fs//1766708:/run/current-system/sw/bin/bash", ":p")) | buffer term:///home/NixOS/coding/takneek26/PS/synapse_fs//1766708:/run/current-system/sw/bin/bash | else | edit term:///home/NixOS/coding/takneek26/PS/synapse_fs//1766708:/run/current-system/sw/bin/bash | endif
if &buftype ==# 'terminal'
  silent file term:///home/NixOS/coding/takneek26/PS/synapse_fs//1766708:/run/current-system/sw/bin/bash
endif
balt /home/NixOS/coding/takneek26/PS/synapse_fs/spp.cpp
let s:l = 18 - ((1 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 18
normal! 048|
wincmd w
argglobal
if bufexists(fnamemodify("term:///home/NixOS/coding/takneek26/PS/synapse_fs//1795284:/run/current-system/sw/bin/bash", ":p")) | buffer term:///home/NixOS/coding/takneek26/PS/synapse_fs//1795284:/run/current-system/sw/bin/bash | else | edit term:///home/NixOS/coding/takneek26/PS/synapse_fs//1795284:/run/current-system/sw/bin/bash | endif
if &buftype ==# 'terminal'
  silent file term:///home/NixOS/coding/takneek26/PS/synapse_fs//1795284:/run/current-system/sw/bin/bash
endif
balt term:///home/NixOS/coding/takneek26/PS/synapse_fs//1766708:/run/current-system/sw/bin/bash
let s:l = 2 - ((1 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 2
normal! 048|
wincmd w
exe 'vert 1resize ' . ((&columns * 150 + 120) / 240)
exe '2resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 2resize ' . ((&columns * 89 + 120) / 240)
exe '3resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 3resize ' . ((&columns * 89 + 120) / 240)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
