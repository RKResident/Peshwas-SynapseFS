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
badd +14 /home/NixOS/coding/takneek26/PS/synapse_fs/spp.cpp
badd +285 /home/NixOS/coding/takneek26/PS/synapse_fs/push.cpp
badd +78 /home/NixOS/coding/takneek26/PS/synapse_fs/pull.cpp
badd +4 /home/NixOS/coding/takneek26/PS/synapse_fs/.gitignore
badd +83 /home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp
badd +92 /nix/store/39l6rhxg9qb07q3862p4zcr2s4146p1r-glibc-2.40-224-dev/include/stdlib.h
badd +242 term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861250:/run/current-system/sw/bin/bash
badd +441 term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861253:/run/current-system/sw/bin/bash
argglobal
%argdel
edit /home/NixOS/coding/takneek26/PS/synapse_fs/pull.cpp
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd w
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
exe '1resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 1resize ' . ((&columns * 149 + 120) / 240)
exe '2resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 2resize ' . ((&columns * 149 + 120) / 240)
exe '3resize ' . ((&lines * 20 + 31) / 63)
exe 'vert 3resize ' . ((&columns * 90 + 120) / 240)
exe '4resize ' . ((&lines * 20 + 31) / 63)
exe 'vert 4resize ' . ((&columns * 90 + 120) / 240)
exe '5resize ' . ((&lines * 19 + 31) / 63)
exe 'vert 5resize ' . ((&columns * 90 + 120) / 240)
argglobal
balt /home/NixOS/coding/takneek26/PS/synapse_fs/push.cpp
let s:l = 74 - ((15 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 74
normal! 049|
wincmd w
argglobal
if bufexists(fnamemodify("/home/NixOS/coding/takneek26/PS/synapse_fs/push.cpp", ":p")) | buffer /home/NixOS/coding/takneek26/PS/synapse_fs/push.cpp | else | edit /home/NixOS/coding/takneek26/PS/synapse_fs/push.cpp | endif
if &buftype ==# 'terminal'
  silent file /home/NixOS/coding/takneek26/PS/synapse_fs/push.cpp
endif
balt /home/NixOS/coding/takneek26/PS/synapse_fs/pull.cpp
let s:l = 285 - ((19 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 285
normal! 08|
wincmd w
argglobal
if bufexists(fnamemodify("/home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp", ":p")) | buffer /home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp | else | edit /home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp | endif
if &buftype ==# 'terminal'
  silent file /home/NixOS/coding/takneek26/PS/synapse_fs/network_common.hpp
endif
let s:l = 82 - ((52 * winheight(0) + 10) / 20)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 82
normal! 01|
wincmd w
argglobal
if bufexists(fnamemodify("term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861250:/run/current-system/sw/bin/bash", ":p")) | buffer term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861250:/run/current-system/sw/bin/bash | else | edit term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861250:/run/current-system/sw/bin/bash | endif
if &buftype ==# 'terminal'
  silent file term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861250:/run/current-system/sw/bin/bash
endif
balt /nix/store/39l6rhxg9qb07q3862p4zcr2s4146p1r-glibc-2.40-224-dev/include/stdlib.h
let s:l = 242 - ((19 * winheight(0) + 10) / 20)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 242
normal! 057|
wincmd w
argglobal
if bufexists(fnamemodify("term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861253:/run/current-system/sw/bin/bash", ":p")) | buffer term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861253:/run/current-system/sw/bin/bash | else | edit term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861253:/run/current-system/sw/bin/bash | endif
if &buftype ==# 'terminal'
  silent file term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861253:/run/current-system/sw/bin/bash
endif
balt term:///home/NixOS/coding/takneek26/PS/synapse_fs//1861250:/run/current-system/sw/bin/bash
let s:l = 790 - ((13 * winheight(0) + 9) / 19)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 790
normal! 010|
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 1resize ' . ((&columns * 149 + 120) / 240)
exe '2resize ' . ((&lines * 30 + 31) / 63)
exe 'vert 2resize ' . ((&columns * 149 + 120) / 240)
exe '3resize ' . ((&lines * 20 + 31) / 63)
exe 'vert 3resize ' . ((&columns * 90 + 120) / 240)
exe '4resize ' . ((&lines * 20 + 31) / 63)
exe 'vert 4resize ' . ((&columns * 90 + 120) / 240)
exe '5resize ' . ((&lines * 19 + 31) / 63)
exe 'vert 5resize ' . ((&columns * 90 + 120) / 240)
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
