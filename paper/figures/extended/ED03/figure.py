from __future__ import annotations

import base64
import io
import zlib
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# === DATA SECTOR ===
CRC_B85 = r"""c-qySYm*$ud8I%1{+NoqUp{InSYn7Gv=nVw`vt+3DOnK`W$R+c?yukTyxG7EFx}Nv)m4q5y=zg!0V=a9^St+S{`TFU|Km^JpZ@l{@BZ}tyMO%o^yk0)+xLGu{r<0i`G4R4@Xr3ryYK$~&i?i3zrXvh@Bj4u4}bXkzrXuG-~I4!&;R?M{_@}a>1RLr_dovO-}%w+|MHK&{_*?2+TZoJ(+~gmKmYN^zv(ah`*(l;$KOtW`2Gj|@Xvqw)Azsoho1kp|M=4%_3ys_{@?y~`mc9?{o%XQ@5UPR-^PFb=Se?(_qRX(;fL@3^zQe6{qA>v|KneN_#eOWUwre$DTdP*Z-4o#x4-`K+b`dK{qCDDfBLgGr?1|A@sqFKeew3|Z{Pgux9`4w^X+fme)EgdSKs>3PHu81{iptKOey5kuYU5)w_obFfBxmqe*W&CzIpS%{`t+<U;OsnZ@&Ecr*D6A^3J(aJ{32`5@YnP<&<JRr8t#ZONc497PC+K%{+zbQY*d|=VQ-$R8F;rkYe{S)YiIw<ct2w8eQ@srBZrZzmcDO_0_wt^*cZP!glk=AAWyIp~Y_h;_d(Z^_y@0<;_pezwz#uZ-4sc*YEccYn?)lxmRtcg<ejn`Kf7f#rs@x?)ICL{=f67JMW7RdPr?4)!bvtxwX{W$xm+my7k)Mg<Q_BU;m{4TcT^HfBI)F{8vB!?XSP&kA3{=Q<{1WAqH1_cDZ})bxJ+Blw2w~c;8MfPFj7}w{gL{8j_Y=dTPmc{m6CwQG4}Xf2`;1V{|dMQrB-mi+-2&O2*>#Pn@RQ-0T)&or*rUkGTe?*Ut`gY+BU8DJ|;2oLZQwYu=|4v<IC`Ub;S$E3UfOOU%ymv3u94(|aiT3!N+LH=rfI^G<K*ymtK~SE1kBLGvNj=)Ei1_o{=Fr>K+AW8@m>E0j|!Qwybr=z6O;H7;B2+EXd6S6{k55}(0ohpr@jiBxM!e#1%9vft?lwD%j}pKUwy`$$!<S^Kbqt%Z4A?KB15(yG@Vd+;ghc;+cL-*rt=@HJ}5E$S5U{c`c$C4S^X(f!rEtI0*}dCOtaqTeN_TX<eR|IB9QZ$wN})*B8^OVMF>_AQETr&gR=FG8qO_fsyq<Ha|9wN^Pi`hr@skNR^(=fB%|@~M`jzfSK&SQgiB;%9Gc2j``OMf0oIYZnBCpu3=-_!9idcWqwBqZglY(6z0)A!%y*95JQ}-fHzaQ@Xb;#NJ}ED;`~FIp``BowXkF)8hZ>o45b_&DZbVeElzPzIywsM;L^{%WS%?!v3Cm(aH2}(t?GoLi<wnin$KDK!T&ByWg^lW!@kx+s2;eAp1JGT=Y6+$65&CnLUVEFFYGG30Yf6I$mA>oAuH@vIonSy^}S_NtRfaOJM3slL6MRCcXI9T&hj?xpcE&z4_Mlsr1!KuJhJl*#>s91zm45Y73zUUEWYmx=6aUsmZjt>~t-8;rgHee{0%mt+S?J+3I&Q1lh|R3q^Et9Vie^y~-R1{a$LW2mP9z7=eyJHq@%ypXSZL!UMFU5olA;rwcv^l>|BZE2D0>oH@a3j@nJ)o$KWbF+*&+<9vyEGqCU!?Pdg;u13oGnbzIC)_3YzU#kdJ<(LX>gY5yIL;KO*WWLk95m>f`JxqXHxSFG^li)zhlkcD#!m;nZ3Mix8JMgo+R(mXhC;kHa(RC~E<u8Bovp4VFe*M*NwURG?@xbsP_(@;J>4xi*$gksoc_;fTkD`af!4c{dIe%U6Jr`a6c}uWvE4u~=iffW~))l`9Ky~P!EP?_A;jHIa$;H#=tL%r)m9Xh!i?D9_d&LPlE^?eBA(-4}fxh<!Usdn7JB2a@%aDtrpMj*@fG}?w*6m~`6CqznPMl&$-BbA(+DMR%NZON5j$8|G9+H4ofU1|qC6TW*Yb4g~W51|DH(w`<cv7!e-z1rz#9h&m*Qb+%5$rh0_mDl;d6JoMH^PRM$7@HsQ6`0JIyW+N3f&Zndc_k0->7$^4<xU|A~}I=jzRZZ2;j|>%-D^UM{GCy;U>lS@<p=nm0P1P<EP5S^O+kZXKI|Xn!NlHS$kQ`FmFKC?PU)`qOT+$qk*f6umwgZnTd&Gm9yZt_72Ev@kIy_WE*Aa2OwXyBfot6&98p`_N%u)`|U&XOIe^R*CQ(^(k^7nb@H;ljEr}1JV>`E%3JZAsiw0wZ$cLCV(;Wqj8#ahm?sS1{Jrha!BZ$_p>%Dr5J^-rPBn**ZO6iO?-fpf{`v=&6Np5;5HHo375Nvu*=ErMsi)ew_f<wO&D)HHJJ`us$PFxAc3FVag#9(!J=RSS)C8R$KQ3*R-787oNr8rbKA&A$xP$%7M4xmys~0dfyOEY&)s<~_TV>5HqaiZ_BI|sHO!8pfN-RA{`<aKTcUp2Hgs4Ir-3YIp>6^+^5>o0GTWs32;3&Va0;8_BdGoOJH0@>=6wT>=>JF63D^+~gb!&8`b&GY~lP?CSa!v$7I!Z2K2xra0!aeL^52{z}S&BIwwUDWgL1B@gM~;YqwOA5HyV03Ry?VXvY98$sd!VzpwcrrsXfCq{K2DKrfE))Gb+Ai3bw#L_b0PAi#voT?a<bSF|1~sc=~f}0LXIBv@3p~Vi2_Bog+u~=ls;tMu#*c$L3dSne$yxvPiP%<Otc)Qm{hK|Vo+^Dt}Fq>vv?vYD3xzfMyQP`l$$7Ma2&BGo3LWt7mb22W%d-+I0^m&2_q-?)rra4L(>+rC9(n0Q`0>LB}<bRl=Fb2;G$l>{+1TW;Fwj=j<gm1A)TYLeg~J0g8qWJ9TXYjNb0C+xg8@|bPGbT40qE-(6w%|6OPj)XD$n*x+TfF?u<+Zb*(tk+*-SKCo3Mw8ke|!3)`%Mu(Jvwb!xiqzps@TT2|L9YF3crldPD0B9a)|QL3J|ZeS|3l(F_)_wL3=kR8Ay;Xf5u^bH4TqfyXXZgMZY+);sf)1U9;D|K$<QdGVFsJkv-uLqK?zV<XA18c3<t<}cfv%x8;%&JclC@~5a+J?il%_iu@SL#)YD@z~)pc^8GO$JPtQztxQo@K<fcpWIf$ts<?7Ad+c2TET9z~W2j*Ja>4t}fM-^?O)r6Lb+KxtV&Rj9Lx)4ktZZcU6IaejZ6T$#`pzGF3QlzNH>5x7X#t>qj-p?CDRkCq>}`P`oluS$qbi+e!|1nFP6Yp+s3HSt@NsI}4aC%&$y%26nj-f~X<`EHKdVAtaf%37d-!n=L}q-yqwTg_=>YS)MGyNWEG{K#^RNVXokj{9OOBLAayju-+VyE!N3!_;y{sz5*7VD3UXJpBbR6=3qB0Ll*|gr?aT@sx`Q$;IQ5nfTQyI@tGrKf2){OeE`CLfxW_JD;H+naeWsyG|E0R#^A`J!%I_8^~bubR?~mc1qFZFlu5Ga2+E{|VG4BF$&NLT7ct`F>!#q?qQeD4AjH%O(reVQA~I8O5GDovgvMH)%Wa(dE{mokr5CPGmgX(Np*4q>b|6iW|Kb(w6XT4FAYWPWvF<A6XFo~~oit_lY@qphLvT#Z;er{Ed#_8X;5Uvgk)HI|1(^Y@NwIMwp2rFVZjSFXHpCGqU$+AHl^pJ}1~Oo=X&(x=6z`DJ3W3vuOV<?a%M9yLR;rW7MCXjwZa&_eHMqO*u+=K)PE5iY9TVM#LN@6sPT+!9VUVR(Dc2EX3yyWWJ~j(G)EzcN3tD)`SQSs$nWNPm$Yp4Mx(RYjt?pp8TuzY3zwmSRVUMcAYCDm;cIGp=32dF^!sKxXcICEeue!?SUC4PQ;X#!#8D!0viAR(jE<_F<4?E|0_OTWIkk+d*A;rQ4GELSqrg%=MDl?*QHR`+94aQ@t4i^lEmuIT0S~K|-N~QX^lh2(74f)>AypKj3N1{KGF1S8#HXc%Pcws(@!U#N9t(CyJamT~(S7j1v6n0q`C7pfNz9Gj?2p8t<$FU`cOLim?(g}h+0F?D_PEmJ9aHMlY*Jm0XV_JIQ&qTV-1O&5&WMj?YzIZ~fS|37ToSAc3f|Kv^ND3v?9B5@4bw)OuqIF`$a4W~Oz3#9*#Y7D*1RQh)&+&PRsj%kJ)GGHp3V5B>w%jKH#XVJr4G~3|Le*MzKl|{%z3Z}VS#DG8WWda!)%A(3_Cn4^QQn-zxU1%{DWq`SMS-g$jXDwnMUd%OUe^W9qw3~8g%-yt85x1=w&KvD!-be4$ko$+bh`n`R_z;55|y1uog-34Cl-gs%i+j`tMfME;F`lr<DlSS()rfsp<6Z5nwMoSg#CO6fzjys;lPl=ZzP#|x)D_zTXJ|AR%9KEOe<5O7G>62SBBPB83KBQMmH9HK^^4OpY^59n}vf)4j1DJH-(^e)1ySL;-uXPp%l_J9eawp$>NIKbv>$Fsn~;Mi?5git><NS?B#wwF$Hd!9pDeV|Ns1-RnNRsEEbO4#%x)gDJ42#^*vjSe01Gj9h=axHJ$D?>P-3{!Wwx8aRkG}!k&I}UBBmDM#~m_39seo#bxhGOKeGiCZhuAV(9Bs<tY}eOP^gRqF<41msJ#A==>SR$-$SYq_rhiInVsaRqHmAkA}}I#9y_GO_gHBQ%Rw14Ykr8;43&Tw4({aQvfgi)bRjPB=Jmdku{nM?Sv-6NisuqwBqRX$?K_d0t$FuR;_*;V9V>3x2vnJ^QAB<3$U9ot6SkW5Ytu)ll*(dT{NqeG`ikv&VeM0_N{Mc#ntz@c7?&hO~394ZNzHt3)!7YLZ|87hH9{uoeof@%I)K5Uy%{9GsXmt<$Lrm0fe42rATw{so4}o_L<a{-bPw?ezpNMH7}hMAyl9=8Y5#X-q96K%1WPL)ahk|7z)u2gvI7*$$im&nWD(4`g6O4ykU+Jo#jvE)nyx4i_${eNv>9Lp7IdMSgmKuW;v|K*U6$}M{`<KpeviI|EY%3I^num+CjxxB!eRFSYM2GkL2~4o?YEfQ&-1h^vx&kiO?nKHt7J<4yv=CpAcGxkN;;@Sz!*Zs4{Irz~EP{cW;2E!$+v62(u}CNzQuq<U|Y5^n*ZQYUJvLa&#!sg%NHb^vzA|gwF!FrrRPU(>u>OCQGmC>PG^WM6G`$>UA(_WYZsNZG`&Q0s7|R_k!nE34y5xM%g_^ki2KGibSO1#Yk4zuz^w{?Ib#5`jhkceEa<Dh|YmMH7<FI^c<ah!AA))FL=p1=y4qQ;2rIT3gG$}>A8;1x6jUQ*xV=fL4ymuP@WBiNzn@p&{4KL84xQ9?uD#{u1kJ@dah#g&E@Za%t<a(qbge*L5yP>EK=R^{Fcm$HOqIh4}s(a_~UON^R=YV@d&;6$vYx-QAj1^O8A}5nn&LQb<o+S+sypBkQDe9x;6DO)5k!)c-aSd)RC4sCRX@Kv2ajsYQxcMrCAg%kjZpY<4-g0G7s2`H?x;xt=FINf5@u=ZU!$zG5Q8Fo>Yx3b<V^uiQ7isew_#I#hcj8zxFh)X9nqXut<?m$H+9fG8onARkZAE&=Yi)6a}?zT@o_@zWiM62;{El3soI+#g)1e)})`SoPlnmzrr8xgB*guohTeWcLT?lAE~|YyecxVoWCq)q#K2Dv6kL5t#mC}Xrnzad!0$ortIrHnlIkO4ydkg&>0^}z<TR)40Y-!`gv*m)2FEUy)}(As`MC>0BauBUo7-^Jiq<OSLSlxUht2>=f1s**X(rPUj3`~x)&UwSMGK%I6kl1@4mhM*XVcyy${b^FMQ|)ln%KsK<NfGMGM?AL5IHiNnYKAr|tAoe6uOQ4-JgElcD?358F7yaxrS`)MlQ+^uje?0%~F9!h&y`I{iVQEuTFw<gQ~VH;%GJ4f=eI-=d+8mT<;ovyBR+LW`-dI=wlJkv}F)xq$S>rdU^g@yl2(8!!uYnK~AhuN4NY68&c->Gg&Z8->uw?=Y(m>4_5pTZfRP;uF&ay|hs!(pS(+Xq}>Q?RnkeuRwJRH;_(mws+h!**pQ87Tc!a^hvUCPB0b?4J$k|fflqlMhJk1f|=AOkfj=xx*%89a9`W3VdW{hg4O-T2BLnqCTno$OeY2F)?lbirk@BPF`1ky%viE4Fh7K5Bl^uUsJa@|+RuTGU^)_lGOH$gyYl>81MB(1bx)Jc>*W$?RnQ5d1yH6(3ua`XG4$#()zQ**&vl~0pw7t(-nyb?MGnFBQ3H{U?7b1B>Q1z>ZU?L3x<p|No-zKC42%9~VI&A|N?uBdRj!y!eU~-VGU*biqlukCUE2Pfu7NdQT5oCP`{<j<{5*g84^Ck(dgP8!EuWMaDbZ;l+vV!>u0nDpqhYlvC}-BNFbohr0qRXT?7LtZ8V`yrS$nW{32W_SnB2+xOL2J5@&OyY&r!asg2HkGrZ;A>SA(>!7)d7`T=|x*P$LSUT9UV^XvOJ8XPZ6J`mMDkKp=4or?;lD*P}F@ERc3^JapaU9|;Qcggs<vAt{(0BZLrF>AlAcV@&KffO>N#dkah}_Lg<`I)s$zbm$BH2ZcF)IRHJJ%Zv~f(8H#so;x|OqxAMf_F9ZqkfT>!ZLF3F)L;&EZ4UZ8`k@BbB0(s@rYud0?)~7`RgB(0ll=mrBXd&n_A=f3da+tlBE4C$f{ZD63%q207~!Xtg}sy;2>qE-*{dGCBVv~hHKHz-(1R|w_{_O>nB;1=ui_$EJ5$vE3;q+#zAK%5r;9Bc0(%aDs<t5PR*0S&wfsP!TAeY4HpWx}<S6;v?KcPBcV@KLd)o4=nX2wIv(osfjTPcY>ywseJvhDKn$qr>V4;mM+yL==Q`+mnyj25^M5w0Es`ubym~><s+@_u&!A=t-eq3hPy?562I)~+VC$%?Vx=XlvI`)ZHSALTFM1mEjI|5AW4~2D;84zc4n6A%b`u$1m)&939<)AxUGhHw`_Z|igSx|<2VT(&6tIgOu3|d@se*@Z&OKZP`_@L0=G3o@#FMJfIs9h%8kz@MRI>2lUr&O3hEX{QT;vb#XUiQpa=7w*}Yrj4}d~0U=l{n&?Q`@h`6W^ZQekHE>9m(z2;fo)T-F}2K-ot0eZZCW2rD+<n&^NKVJHD{~NBe~=gc4bsl80!}RZj)|Lk-d<dV4gIfd+LV47{FF<VOtg3zvKetsnBPF2SpCav^hDm6<3-7q&onE3nCN0#AjP40rqFp&0LtNqdGcQW<nfOu#gmjVKf-#=cb<U}79x)<V9}x=n1W9|E#WJPJ2U(8>12N{S&es@C;O5a-KkP4Jk~63bj<7-KM*`kd~u6Q~+MX(t?spsOTojw`?Z)&gRI+pN;E3*}sfV=i{qtMgkJ0CUmiBX#ea(ZVp_IYwYg2y}*5`ap&@L&XNhj+qlW_p<UJZ7U(_*ii6f`2t3>39cFaf)<zL^y+=uttj+a>l{UA7n8JfXUn0}zm;qp*GC=0ju+KsyS0gRTUd*;iDl`+H%7P|VMxk(f^Lg5)i@sUlv9~frwtzl4`*%pJ)w0FAoF2($#Awp$0ct>AR|jypD+E<Y3U`8+!JHT2~a9TT|-Co8OB0=3=N*0Sf=6iPGN8YSp2#LAA_))^2qn#Yn_a<O7}hMstce9B6ZTEy*k}xt65fNyNMYqOf7L9U=M1Fz67sbrm{!VI*8G3b|lgvDyX<deN@b#*7I6!<33m&a|U0J>x#aBuBp-zplCs!Xk=bkp%U~l9yOdsp>oVg>Qcw-;EIL)Bj146qxzyRfoq=Ntcn;o`Ql8G#Wv+?$wP>@TCc-I6FC7#LN4ELfa~u0<O|SBvy6+3Vh4t_A!n4IYYmH0I7+vHxv=tE6D@%>1zIX`4q87~K6%ZfcZ6CY#1y?N=#A+98lB}CYIQ*5`eX}&3k?IfJJ8LS_zBeBl}>)(d)pg&JtLX~)s@~Q6PkgypBY_pzO7ry8Qe`El|zH6&qMFQZOs=TxIq=Adn_Eu9uh}kTAZHeb&O&{0Xljd4tBxR&4SE8@bP`k7oa%wN#Fo$sS~y_C*84pI=b$h^Bt@K<V;9Jxaf@Xeh!F_Zft%D$lZung07!JCK_L~*G9u1q}GZkp<?EHSeGb$R*J65Ag7$c<HM4`FL5~*L5dE$gGmbgBK>8_#yGo8p`SCpjV}`&4d=Q|b}!~|`Qb_6MbCU~PVcrX@FU*TkHGCcsxA3Ze(!;O$q#dU59>^RnCH8H_Vy93FOcx??CrvbUI5vWmSAjz(NJihV~0{qI<?X#eMFSG91ZGEb++iekMOMwZM|a5Vdx#U{2XH!u6TctUHIUOwK4gniyj?xf*|!a0tWf6{@niTkUi~*T0lh0NuE?4!UF=LQP6cvEYL2%kc*Yt9q*aiS#X)!UtPD6YgICTqr}8bsoI53Hr%r|>;{3F{1B8wv+)FX?mZkHDIz#JL<eXb)m1^yHH;EX5|-iYDh8t4Bd$CH8>?i6N^9(1E~=cP$ox|5@EYlT!l0Opi;3?Gsj(cz*3r%yB=PiTfT~U#w7y~}wd5+&iZ87FFaxsmIBhGGF|(rJu{wA><oAVhA2EtEK5*A6o{Qd!0F!R@1f}JeW3k4iY~2Aw95#v5sj|w6X9%gKm5{X>ZCjY&>}ps5;i%BI%PP=ZLN^i&mPPIdOmI3x72dCAl*~N^^lDoV#`ZxMlis??fKj&(&)@xn6S#{WxhKpPYrO7AsEkte`*w!tB4cDj!n6dsjyy4R`zWcfXuZdP?6yAO?GPIn)T)^(JTFY=UVpQqZF5mGYJYxkJuw<8r!~RtXP)33!fx&dUX8EJKo=NWQH?Y5Z^W7cT%t^A^05tGBZ7+%yP}1816>bm2EKr;DA3Sw;4o+<>nF8L5h`uF>hV3(mRu&jhNTV$XY&lS9@Pze5m;p$pzPDVl1IbLCicQZE$u4O3Tc>VeTnR3s6Fr9SnuY5b^rYBOIWRLvTmYAJBZe20~^8Hv||!R<TI+)0(CHNy4k?bby)pO$=g+r-VbIgWP1?fS#%r1bDZT{8;dE)TfHUEC?j$?pP=k#Ox><}{N9djogk=7XqubYQ9MP3xbW0Px?9>%hb__d$SfC~9?^#zsC#z<?^^Wb?#Tf}Dl!RcW3Wga#Tvveko2_lz0j?~<s5yr=C>R0dwBou1sJ9$(uS35!t}L~yzL!j;ly8C(%?RZ0uwsU2yWfN;iEfvFW_+}Q-b<DULe!WI%`=!l4VD}qn(v|5h^{CZPXzNJ%`4}Wp7`iGVCA<hEXX)6w;p*)ahm<&(xp7kaF%t)zv^vqL2fLv#9*&)a|lozA|_Bb7XFpJ@-f9@NVzBeH9+>aZR_6a(NHyx_z9_dqC#)Xs0)}OHyGM#KU$4*riXsG$K*`0*HlDUEV-xa6;?jpLE%Ej01&2Ppa{W!%eT)nnpRY-GzqMSIs!PA1miBT=M=PyY$JIVD>RjzKo1sSiR`!NceG3K}RHu6WsA76>@qIWb{UD4&pS28fUgBgqGO?=Mib;<e(ZrB300nI&*b!YxKDG<6XhXc7wEB)|x~liOFJ&E#^ieS);S%s22d6HBAY9BovV{LMelb*wp3dnf1wp34`P026#5m-}*otzxW(%gh>DpKyw~xPU?pyC9l6k|BV0aQnE3{3`vOoQg}e>;NB@NpdJ=wkMnuDuXT!$=vWB<S8ib~Ov>gUq!Y3c7&4w9AcTpkqj}nO>raYsCfW%CBosq-N_%J~b=*u=$UrjVG@!;VZ0)T-Tlc34jYnYQ?PtC+guK1rWAO3zbB{sC+t2+`VB><%_S%tg!3lWXxOjU5uMrimB^Dlmj?16DA4=Aip;c5U7LA`ZKSN0*+mj7=R@jb0pv}rF;!K~Q<no0dgO<yme++6afBx?1S;)QUBl2TKa?z%-(1}S*kS2t4jCTPtiRvKcr9OKDK39Ct{a`dDTBAq?pc9SOTMxpQ04ZBHo#;Fyu5zuk-Fjh?0qY!!t~eRHp=q}n>59Bqs4<jgCc<Y0usP9>kgV;t=wByBn;bfuqks51nl9hS{irG|(1~$Q2ZScQaLP$%%4_|I!1&<KJwfP=nBAtCdmdI_Bn}>drwbmsqq|u;$@rR$*H%dS1f&pn=OOCZOqvu0Lx(<Q{{%=EtoRrNUGU&zpmV{4_xCu3G#xS#`UInHFTYbWEsv0^!Z~3!&l$livN|POjbaWnmwdV%eNK8!<y+PYR0!UMmU(J&dK_|$p$r)y3Y^+OK^@4+7~DY1C8uCVr!!76SiQQ|z#jCi%Y<4T9h)xclBD5!R0NOYl+qV9kCF>Eu!Ga7g$Mm01+<i{`KiQ7t1ddAM&BEUFp!|ZSY3n?##eF$AGuVIPgyQ~>ZOK$eXH9DiOQXf1y!>sGp_L41&FPh%!Oj^{7kU9df0phnXC+?2R~E5&bQ}80ZW&>KSVBl^8OgP^vPF>Ibk80O~v3VW<qT<>W=dAD9PfOAn;Zk{5ok0<mr*$tL?1s3|CCA@?+#I^g6`r_?-I2`d?iMM(Ngmz>Qt`K|!l79P=RZzl7*GkpzS^SvD%<0?<>?{be8xg<7bq2Vc7lo(vhX63o>My2<c|J1X!r`ma3<8~gGLD(tdEp2AEaVudJ1QI}vFh548&s-3uGGHJ~;VOtL8dH)zwk*h{Kw*}}0<y7?58rrvM?Wb4`m&`gfn_~(xPrjdWu9HWsL4co1ibIeW8e^Bt2wOIXrhT$MLjIt4Zc%z=#10~(1lQN|l|MRnx$K#*3@SfI?sD04kAcapxyx4wlH2o_uLmOUNMODefIJ|D`4D*gTuIDz&)yF%b(ksjIK9Bo&`*Abmkshg=SC61`(!N=`T#U9`i(0;1~)%v7IWS6cgIlyP{<jzc<(A~j=(7MF1jm7-0QcD6BGR)>UrGM(Y84Yqxa`Ax4<c4MlN>v>{10p1*f{M5#<kpLAZTQH6uGkkWC?b_Su-^=HPUPOy+u6)w&&hP>E!N^9p`xpVHfhQc<C7uqpLIhBndFl@kW!2C5#I%6y5gq00&gDD{Fn;rdBtZOREMUl+)bn~1NALfU}#QEg*SGwAy8T;{@u?ue&I`vq%l)yHRY`~#krO0{T=Eu;L`^q!lg)?EGsPd`gCbK!%JLD0>~%zIo=KQZl?JU(cFSrAU~O6eAXPuHPP0B+D<ayf>egYMl;?7SnNxyc6=XcoF#wMs23>X*VKfE1=Qr_ES4Gm<N;p<8*&+BfD9^PX(xdZ-M*2;!F>7EEOB1RhgiO1Kvsyby-kP>9Ach)m4L$j)Kq5edzgAekcCXsvf0nke62AgQa2>L?V0g2};r>MH?Eg{iXJK=4bD43>$#lv?=eElc_0?bqMF`PXmXef{R!-@N_i7cYPkxwF3PF0`T-PJczS%g`Ird<wGg|K$9`NNE}6Ui2%W!&MN69w`k!-E(KcD31!|3RK9(grNPb-^V3QM_$uy^wj}^mBZ;A`QVux&>@%-9J(3RIOtQ~SCS&aLyGn-lnL@1VvjUXdqNol-?i($1i(ed_?9T5(FrBp;23C*ZgL6~zR)UNm<t+*pB}4*$|$OP76vo4tsQZM+!!d*mr^Vik>+cWT>1JvT%o_7Zv%%x2qOHbWiXFW&`+fp;W83tS<#qqlqi_PMFJnPO6$TGsOrw^xI0V!(?Xr7j&c*4F}kkb!xgho(~YaFcTArPVrC3p6fL-+l&EXu`3~9U%)5>}?7Z4WAAR6PFl5LtAI=DR^+;Hn!C=%HXvSZCn6`D~7dZZ93ZNHpc=x7{tT!ru6C&uwMILuThp7YymDrdlP}GdJC0U9d2JOH8o{XMYhg-dG*u};QSCylgD+(lb3-p@xrvh#E4gpJCWV;mF1H9dk!VI;!Mj*gy5&y`R8<I`aO@~+vMs;mz{Vvv81us9sG5ZG+p*ZJ_JKYCjGPx32!HjMMK%})bIeoC++Z<1q`@yY`)l|eRFN7Q*r7jA7yp*R6e40~vfL$<Ba_V=K5ED?T@1SK-#RWZ+H8fPLHlCD%1)ZvSyRd8*yIO_RryA`(pnfis{J|<Ha;|J-<mg&0dRy{490XY(S%qb*-pMBPYRawXl~$5laCDe-URAW<8LTmzpr8xkLLAqqz&Om?gk?L}$s#mA4Y7%W1}4F81nikR(LOmRXX%cVbnX+nl0$psx;<F7{=KXLlwcflJNvv4$X^8S`mLCl3K}q4VQVykpJLKIv&LZI8QRMfRK+sH<#O1F*iqT=q(kM1+Ay3u>lGt%Bd5A)^CcV0<cuj;c#8Hi1bPJ%%;Jr10KL;V7{}#%2HJ(VsdkhQM$+|Wq^u%lGj9o&ZDJQYkR}MhA`@eR8PWj_C35XAn~=#+a9}XpGqR{-MAm5D4qVFuJ-{9W>o{+SFe9y6!%3f4_+Ww?Em{82=_5kj<c$amRjWE{5Y}yHPs1PwZpw6?5!x#-dulw3P@7RzFvY3{2Ba>h3Ap;$FsxhsP6h&JsL<@GpEF1Op0Th|Zwd(=cw-#U&J^dsl+?~NtJm$rx{d5)CyG{?nSV_MjSH>afR;@ZDCu$<olG98y{NsyNmbjtrC7I<{o)7olv>I3%FFdj_K`U`=#WFL(MIkXjUeXqX=Aj+#*E2WdAfEp9s0v^mW(Y__%-^=kXTQQR4WKIE)W{wwA#kBZz*lwaI8FDdl?Wxm|_Hd53RLXyylC1^W+qTl39Wy(}blIYe!)&9f18z*>=@_tlP^Tb_AUh=QAEv6a;7y<eg~d)8)xI4QE960Kv4b5WBV;!Ng16ga>4qsEH_)(Cr{0-pwI_<kOL;6ebxfiq4swbUH{YwkFRSkA>UVHJm_}Ne*A^2({#AH0L@{4JX&aiDac8@EA_s^0C!exa_@hOpuc{C=x1B8p1O%0y_@U1hoG~lGr8rf8N194k7Toy;!(~y;4ljR!1%|+#!ufRPRL5TcrA=&kzQ`YP7OvakAg^TFe`Yg<IImLdfweFgqaB)vY4vt1d(!48rsFqcj8pLsh}ucb2f6F%U}+(r%U^Pk|WOQWiz=sHv2h=8{0SQVq1wl*D|Z+>mh9<GgKHdX#ocFLfPk(vlI4+en<Duf^qPneO&7rVblb6FFFN*U4(m+k}OC*uxme7UQ()DinQ|N|iifjuqokC)d$2#aQ{|Y!UH6=4RrIJt%2&*+I{Ai)9~m3uz(gk!lD^X7VbEr~kxb$B@AbOl3u~0>)8VhGaxU?JjtP<7yeyS-_j_IT4ii1=+#1Te(#AbzuqPx=zicBSA71(s^AA7?WvHU@Z!Jo+el#y1irUr|64;2WWu#?w?tN6^p)X5>mf$ijvN5(d*9%<;mPk<yNPu>ny-VQ}Mhr<G77@NXM$>L3`eXnKqmosZ$Nlh>xbTM9`PE9HLD|fxtjfSv1D3)V+aOoMf$Z=7XbH>&Gx0eyKnvG&CN|Fbl*E+IR+9Egp?Rwy4ijICr_G^}E<+7rFuk)F?yOyNCDt)H=mCf=-Cdq23Ea7_=b-+5A(kk1Gy!&7iWSX$yZW#olE~Hk_r6JU}`+vZM{u6{tzU51@d7@F3_jR&QdQ<Y8bcOS0m97zGrSqy9|mshifFRlaCHL#IGrtZz6=TgPgu5FXYjA!IdLC4e<*#f6bwL@M>8R#BnV;yUu@*^R8dB5nF&92u#RY^sbzQ7i58_GB~G?_;%P=uK~wF(z5DM0rXtAWk;$>GY0!!i5V-))_HiZ!=HBFoeOZkNT^#9AL<_6hzqPwT<%ZD{QMd+-DT(RA9Ao)v~QHc=4A)KuC2F18p~01`kF6M~`Y5O{#NNVRPMKvrWiomvppS`HyGno`8;Qf?Z=dM4ciKTB?}qkFCNzWrr;`p_?kLT#7&m5l)Z71RcT#6sp+EplGLOrqRqM2cyl}gahji7cBy93yJ1mdDOIx8%LRuT~Wi8-(An+ePb3bp#?ddyXzL=$lAk8dtl>;+JP>x$5}2$DWQ8ngC}1ijV)w4&Wb)1JsYXa+k``l4;L(g{>`L=;7PC~ey&Y~DKYBDFt@PieEL0N?qu>6-?e#@aA@)2l0k@RqWj3Xrnf0<(S4{6jWakc`BwETKcpM%VRc8`yQg`Ba8UK(r8$80luSJG!YBiEaG6p`()CWJG;P)l*INKu$-0<<xNZ*aD?Qw07P{QJ%!Gn2GEpWCwFI7WIj|J|JSR#*JZ6ae`k+0Svk7;X9=6(s()|>W`_8Ci!=(5$@~&T^pkhrw1(9;?zR}`PKDG^eR36sbh@#+A0HbT{m<2KT@{l7E!KK01gA#KtG7=tb_R*|#)jm9+?r_0Q)HVtFNn%v^tIEKzO3;A5o7YY9BU_(gs?^ejNj1!yiANM3E`$$B;e$+p({T#1)432m%rtPrlJP3mYNfr!W~zKFKKglc@sPs96|+GeSZ|o*q`Yw(M>4iblP@bHt5;2PTCoJ`aNdP^!|{;f!zBX}Jl=>9OhH7f*gS7gKyX1?Ch23?MQ0C5p*JQy%-fHHs}C2#ilFF@dYqsOBj`^t3>86=d3fqW^;@6BH^p^<qcr$st{Re!<%fH-P2PvB|1Oh9K_O7jMmvmd&;-TyK;4RkkB|@CPBv{XKx~dHS|n@SSqNBGo@|HDnS!GQ+t_tyRMJUX`^0wKRe#tLSOi_qk|VE=2FheDa*lE}?yJt!P|H$)$2d$X24r#8a2#2HxENW;8KlpuB9^BYJ8ry|l!%P5b$ui`d?$S=I6~ld-eMeDfw&l11i4F@yzy{eeb5&4kwT@y;LntSGUQAxm1^C)Y2H*ETY|V~BQi4?Tu;(8aMfAECC5qcSx3)C-h?`I-S|?RPS{!1^s0$Cwgzz}+vMTjbMzcXl4iQtK3LHN8NqXFkj|CNgWV&6a+$XcM^zy%7zH2U`R)k&66Yzn$eqq6=P8TI>l><po$iOCOOcq8xNZ~rV>JU;Jo8fRI4@!M7Zku5B-5jJ5a>JAmz|1*yAVXW{%qP-N4qf9cpeuI6g*IPD7imxMeUn40|PT0i{4g63dSXv@oPe?l6FY?>rm#@<&VbtB)b|L3UMeRk4gPN>9~{Y%EKiiL4?QY=+q7?zx8F{_6r!^<qhk0!IfTk2{akF+|eaxCWIWD(bi|Xe8RKgAUAnvi1tzCm&Y*beGa)l`VTP!5GJkIz&6zO{Ib(EN~rhBx>PftaQl%r3ADwJep;O<U4Q+t4#(@@sISIyA#Wzeeipn#!4tt2nyfxxML$$`Fcd|%`fi{$MU6pOl}d*}%Q*pf7y_w*{@JFNEVJsuIUUNWAvYIC{~ZBLLpevdtxuG~x>K|fu;HN%l~yiL3Yo-OWOa1N%?yTO`jysdq0mJC&6)BtfQ`maG>vN1K}XqD2+tYBPk1W3_CrJmgUt`=2rzAc_0V4BkttB3ji40m=@cqHBpr5&+T}4CQ9=Z|X+vPl+SZM-1G9Cy5okg^TleEc2NyqaPn-_4jVag^C{tyq&v3di>Fe_#a^#{yNv4BQTlg5KH;E43h0;j}v>+^8<e=)fX&<^!G=p_{c<$`n`HZ4+CuQTIU73UFZ3>24K-x8{`zvINm_a6HBGiKV8R}z<O<H$McCwA8pV@|d&<sXzRW96u(0LL{u^k;F?zpA9p|z3AsjcR7a6k_mh!?sGxy=LgVbX<{;2CZLJT$!t#RV;z5{+r`3bgXM;?QW{6Lz5e6^$5V2AmHQF1!HEc>g4Q1CCRGHX=;i-$u`oi-R+~(}-{K;e~<c8_2x7YT*T3&WQa~+A0(=LS!2hvqhR!J&|U1W$+;e#Cjd=k8W^x%pmjQbqiNNc}JuckWECd^5cr2qdwzXpzn{)KtDQ&3Q(e}8=}=lHZ6Q_#7@D6;hABNoo)&1hzCCu(AE!13s7j5$}CmMgTL%LV&A8NxB<E47U?Z3m}z4|A~3#2)FM0!n8j#qM2`8vQ1~NdurSR4_r2nW>jAvU7xk`YI;!<@`)FbDnc`$I#%ybpsHE9xL{_|R{LZrgez^GI1t6ClEymkK^|+&EmFq+%ot87m)vU-$p#$8}@5oW@aURJJmp*)n=*h!4Mn6=qy449tZ#KD4Z%fYnOQYrxY~wLtvbr6s8Av}&`tTC8d(y`8!tlJfDJO-?ls1y?L9tB#2zMG<T+{>PH{5{s2Z|pqf8=X(ySIrSzB0diqmtrlalAJxD83TU`#1^3$GG0d2`E0s_ugMU@e$59(EIQl_QHo=QfuJ)+@?4aKEQQF`luA$rAFFLr!xs2x{Z6hFF(qu@HG97VYDs~50KhLM<c3VyW&eIO*`K*saTno$Z)uBnT{a5{~a!R8*}$`AnWK-`r;DlT=YyqX-qYCtF|C%;GA&_>Vl{0OsA?+VQqEYPA)^W%*1@f2=Q%H2@1A_^?EO;LnMRQ@rJV0;Dw0DLDQpLtW-B<>4NW+`Nrsttcyo?mKnAf+PdXmgKBpJs+p+Cltn$$CXfO-Z%XM@<!F}k-pY>=s%9%v)={~JaSFaQ_`rhEt!bqe5v9_jwwcSyqqGsM6`s*sTn)RLjeavRp0sLN72jJS6R*EY=N=95h8_1Kjh*CxLrPPTgBjEu+KuoJ$_omBY2_i>2G|1GDWDD8;qYP((KIm}GRk_xD6i=vP{%gcl$I_-Jsm%vj2NOr-vsIfrm<aTu+Am<2kX|q8nC;pNTJ{;&mze=l&d!>x=aYWqoX6+G0AqsNNbAz+H@vblN%FZS*zV%yWpFa6j<Z%5z_Oc^WY@*qDSrs)Cn@=E(6sIC-@1cb>H-v&ss$CD?o0h#dN<u!Rbvo?0ZnUO)*0nmOD*_S{th0T5f=tj!C6Hy1zOg=Q9z3i~SZzZ%kuvfoMO`AfhW{8z|jM|Dh8okdaEGRSG%-ZLVmYspol!-kinW0MK#L{X$?#PEc-ioHCEcOU_WDB?AYUQ2hga+mX&4e*>ToOJiSx=Y~`aI`0aF1`{3K|H*q3n2e#1{+0$eYvdTEvl`<JIv<tDeu>R+c|!Z=nRb`A$%%}df9A>~A&(m^+xq<q=Xpx7X5K*O{WIAwu(@@b;I*zmGhjD{0u%KjJZ#o4Uksn2YiEj$gJ<cQIeh+1sq9sc-Vd+!K_};3RR|J0{cj%tyWvNK4{f~g-NBzI=uYhuz`iS$eGhQ!MCd9}xi7Y<XdZWP43nn90Hz<xeRRHLIQ@XbTflv1HhVpQ%gJEOddLSfW~*rhWZYVi;<V0S*^pv%uEG{=`rIIQ9l-C-X0HbF7N-Uu6KZB~klXL*(`52Hg2rZ;S8?_lV!N>Ymm4sCcryC}oGakbH}tWSAB1b!2KDtT;5i+NgGH1=1D`8H5pnL^nZfg;)7ckrU13^*`jHlI1P!B&?CTbU2%T2TL!ndA4xdLSsOUuY8@PU4I{PKQ(@h<HbS7j{(n;uph>lR3hUgZ~LBAF#@bj{+x>e~0ygxdfz3iE<%K_h(&3=6z_|}B>>u|xh=d)jl4}M%i`%zB#VJYoLdEp1-v>)My_wX5V+RGk#X{JUyT!7TsPoT2#R6^={l1&z9WIS!mTzDdI=^owKII?;W>2OUedA*N_;}<S@SJjR+559!1B`joeqv+Ey?7Fa?F*b;y%(a(GtRKhPD0!9y#SMfpv{Kt76`EVjzA({emj-2Q84V`IGDANr&)G(pjh!A9#H=c#34k1SPM;H1fM!GLqbVHfIc^0M2Ud4ugjIhoMnB>&6uXT&E--a3e`w_a*d}Dmu!zIAAj$7hmC!+-;AaeQIl<$TjQXZTy^LPb<Qb7l3rPLyOv`}B0kRQJE;vDvQUAchwe%ouL|JsuJ4^*oO9U&ok#*L4M()q1a{4LMrXT6J^zfeerbDO~<uiHfPafmxGK*-!1GT#GdEE}y!fZ@hr`}Y6Lf;hegsL8P)dEdhlu_vVrHsb4E(Fu(ZisGm`OP{|`V#snr$55Xh{y(2#?YQG{L#7SC6C+_VBt-mA+N$BA`A5du!<*I34>S!e*h^L=17A}p8)KpZ1e-()863frsU+)$uDn|EGs=H1s-*rav7g7B;-<N*vbY#Mw7!FxE_^7z5uOUjsy)6{%gcc>IT6w0q0H3l!R?qr7>13Uqv4}yeAgBimjWo$m@|+e_FA+A{e>Sh$anAF%Q#zlq1+v9gJK}Z@7_)lf4V=2DBcQM7{*9k#GmqRFfsu!K_RT!ltJcCxPdg#;VdD3;&QdcMDl}&mv#K)rg)NeIbr9Cqd~cmcVUrxY>b0(BZ%;V2H>dR9TRE16n^<8hOp5cZAx~zz<zqIGQ{Jx*zh$wAMA6j?i)<1EISI<}@EZf!e$B$PavPJ40`>=~@E6gaC)4*=m$(8Ae3!pkRV!#-h`XD5?&-^*7H!@6q|+mk``BrA!4XqIO&s8@9^~eCaz0+6EyXon$~2sfQE?)A9@i-<<(o55)|B0|*_R?wU|Sp+O!cUwMhegqGPzg%AibZ#nTbhZ|UYcmnt(CPQM<Q4}FH7NHc>k*yPkTSyC=iOoyXgVFRmF2D&_Z$R>43E&IR9J-$XTw${O5IN~l(A+30)RrQ;rlU8P$<5f{0*N`B2S0<%56=KEdgg0$d$;9)AMvJs436(%x!Xs1zT0!Ruf+8|E_M4T-*^Aq?L(YjAmQQ3+l3Fk0J6}o-^N%(Bur%zuK+<0FjmKvPv#=^o%a}P1nKXU#pEBj-ZDDksYA2j*RFVfkX`uTi<Pms5M`P8D2|#V3@oxQCWRafpnMnDw8~$dbcPypRxoCByi63Oml`SrHTr=G@d%yK7=}5odC1t-?c@s5mUOemgqAKWgDB2mAbq1x0}TR7t&SleRQ`-#RA{Abwc(9Wi*+a_yBr77lDenR5vi24w(<~cgxcIEM5L<?MI>~kDP2-<gdz*T>d}5H(n15zT7`Fg2#)fHBD9zEx|-~9N<!P@3xhBU-&UTdjcALY9GYGU+);EX7q+Ac5cPntPMJW=A(zsz#+5K?<U!Fe%#L)Z@XCZbLPHEtmlmE97&F#wVKv-p?T8db6Ub_YwwxgN>*Q-YYVW6*^&`*l8R5l{-XQQul}_(SfsAVr8^mfbkp?7RKGa!R`h&B$iypZr(zc1a1=BkU%J~Utt20fXeCFK}NM!oflH7(*koGeqaTh&yca$x#VIkp39o*<!1Gl+0E6>tkW)12BQpK!-BYC-`9&VuQ);#Vuj7@=ReKq}H=WQ(4kwOq~ddh#Mt&poT8a5d2lD~nmhvjiEAuRgMa3>c16?q8DS2^u6Y&2Kk^O`qXNaA?MIO*WSp99##61f-PHIsZw(4IDvl?vT_y=rsmWk@JMk{aO7M)e|FeSZ#I_fO-#gjOw^z65CR9bBiUNMi!-D*Yf13TJ~>cHn4q@xd7d(HyjXrY!ENNACx<dNqm@q4ZQdONjOXY6+JS>asC@y-Q@6@z65bWPaPakAd9#ns@JjUWfu{kEWFcez9QGubo=I03hgGnY2x+8CZZ|G#dinfZluaxa%Rf(WKi;D;>?n;0Ir41ybd;DA4mpNsW|jR?J;YMRq7L&Oq?tdE5(7?EQpBThE0Ew;Gea@Q&H*OemrcI1;%Ei8C~16|wGg1B(w&<i5mYMe7p^++^^udYJX9C_R#;B3P2EUaiwzM-u&2nyiCKeayk~ajD#w@N7w*&apybFDtOq-$tzD>2onrPqXXtQ;<w6Ne9X59&_;g=v3~qXTCDu_j6=&mp%7K;{9$;=Drs9_qb&4qx|3Fvbm3OfDg#!KEMOk&oAUfE@+GG3bIR|dTC~&`UNnHb_EJ9LQ#_7O{Y9eCN3pMH$fWqp<otjHW5^-GPaJUf_>xk@zBK5CGQWkOP_oRY9I6D%h)J<UMT3C;A?7fX}F*AKVd$KFaT8(P-q2lg2_4z0t^UTz#$>&Xs76MI8c!=<|Fy5Ol@`CR{guRUvDEw%2V?~6L{?*GIIL<$xUNZjsi>PM>772v8K)pM%9tDRC1-NS4)MO*bN@`)L=gsXzD)K)tlIelcZSX{^L0pogwdqis)47eL|+Ay$RN%wv`M-Ix^Ju9pK>6Wz$Xayms^GJc@dSyCwJyYj>~~DO0e<qe={>yAM^C9>x1d*9>(+os)@lxJw-in&ZGp*>8kYd-jNe6T3lUHTa#FW$@1y{%MBe5g>W{nXe2YZ!h>5fV}<OW8m@jbAKG*xZu0Jc4%C11YR{T-d_K!gvDzahDX5T@@MY{lZEcIOmnLd7d}JDM$UB9@^oZ*tF>ffL~Wm-<nnbNgO<yme++6afBx?1nevosC;Im2a=Cu0eo{E@eO8zb5d%jwD`w?iCvFYwk@M)e;(P9gqCVRo9CNuc1_-r!ZACvxF_xIo;jH5*q7<*4B4&OT5xtJ0D~`rqXzIC9$duCwCKS>Qimpwl38vy?(rPst1H@C9tq$JuCutr|mv3YbSd9=FWG1acFBu~0l<181Hb|Xi2f`o$EFC6jt-B}dVVFVI7YT$%K<a{r?&xynIPp;vT#5b(O3^{l*%+O_q{PvAaw$Uv@ClSISnn|yy5Pabfaih-@9%R)Cb_L)Q~&E|cB}%@g5YPTy>S$bYESTv=zMqGQ)~8`gUu!1Zbz?^A6bD#u}q6l<m$IF+NISyUy9L{TU!*Wv-JB0Bl(|$%O$5^H?OmJV_c_Uxr2=MJ1J7eLzH#MWi=W<v%F4EVa%@33`#Co|1Mr9<koOP=4OG4(2)+WKqMIov)+TV1hQ+guHEU!`8qsus~(@WT>8{YjsChZw*j(SeGl)uCD=eu3XaI;q7yDlAH#>@5GtHS-`wIP{PgaOk{XVMX9&4;#rs3#(kJhakxQR^t)jE+;SogC2^UB>I=;QGw+?xNAiOZoRp3PJIGd5eI{*Dl5JM#w*=KWkY#TUi4Z<AVT;FtK)_%TgC|Pfyq%-rIA*X;lsM<VNOi@26LPnuQGOtp=DUf$aERKSDa~)r%RhlwO7+$6Btq_;eG$gfa55h*6jLFE01L=+~PwNz6Fv!AhSI~ep{3(6XD-K*D*|a(YFy$%h*FuigssT>M+E4?vrcA{LD|fIOD6=dQQwG#|V`#EJ2~xNgv;=790raFZ)NzCp?WBVQ8|o)Wc9G7E#g>lnW2`vJLJrx3=O^gViOXfrd|e><Ir5gvo_h>HZp~Z1I(WPzarsKX@wmL@W1#VXyyb&{@pGjv*FAebpsf1*iVWbn><Mk7C!|z#$IUY>st|h3I`caH&BafV@^j`c*FFCj<h(a`c?W`ej<YTztZ<>3+C6bqZryV?yUxA!pK@{NaxB3*LDDS@-64Is9!tH^?$=VZ7)-VP%8q!8hiOjuIrM2twscWt3SVJworlunlb0_bRp3AAwRZ%|;ddOPN?5tzB0IIF$Ksf~@+co^$JR+ZgQy3lE?**R?A3ZSeM=DeF$tC$lEgjB2*}J097S6L$4Qo>=OHsS2dxj!T`qj+j$S9sT$O%3n{|GoVQmct;d3%3byriMAC`)3_ykKoOY(BzgO5Sa&B@Dq98U+AYcA$2=!db<S=s|SfI7c6`C!TU3>#cda^>J?xrLhdWG^?kp7NDx*=O27zOiB&uUm>VX3n_IwdHf)D?Ay&xn?5V^LTk+_VNW#hT0K?t67i^6{e6U3Md0E28B1do8~*~?l}4awBtC+RWnF=MEdfQ^O;P=JTb-`VQ)o>aLO>T>O?uAzd<D0Ww@tutZ6R243Y8cpL~0Ymof@py=66Dy#4yyH~;$WyRYAT`<u7l{Ne>nMo(+bqPsw+h_4Vw$iP-K)X><f9E}byVr=j|;x-c%Gg(^)86_SBN|GL;CufORq<uO8w7#n1Aebc9?&JdY%4Ws|1SpE_CmAAWej}^~<)rI4AzfEHoHqHHsDa`wK-c6*xg5FUQxwMY`+(oI>%Pc@P5e50j8I2~LRYpSSDhT*z0icyg1D%cU%)UxH@f)@->=P)7h0p8vENWCJ9C>NN|~7`uGKhn*KXlr8-BW>>X;k+AzF<I8Zy<-uAC853N&5fFDQk1{zgBz{GVve#x?7TC<<$ohPr`}j#EZsSbiL;h^epN!xg)b+f8Z8A|HnGccqFGZPrOgBqAN`p=p&1K>0Qi5!ZqI9Y4(nZN%oZ?FKL;2jON9u?e9z9HfoKEbSM1bYUsJg)V^1fnK|Ap|$H3=gb_yu-;=W^f0tIC_P}%uN<7%lTliE6RLBx1R#gz{f5JI%_PiIdh2&|=mRA=9Si+~T`0@)C`j>)KERP}dU_D0Q6Tqz$ZsKFfUc-s!Z6lyyE;tr&cXF#q~vS2vDPqn`Sm&LN+;b|KZdp<6>VXwzKO1@046v^Wb!c-zi_rf9g_rFpohUY;4YeUUgb|9SK=vl`)BL_G@tSSvj8eU`y&5{!_qp=tn99?2&54XqitiM#wQfI>646L?7CT4wvRoHLiIXNs2Cb`SgKDp0o2h=FnV>qLAH4cUckpDVcD8@G6)%W4L1}<j#%NUui6q6UwV~Z#;cz*qA4P#EoT;Dp?Qn2YzKSUgMzdWQml+r30q0VPqJD~?oO0Z@HUUq2;k-B=^D@>cf%em+rfU;AoWR(D)xYmFMMX%UnYGkYlO|-j!jNb_oH!BWd-`IL0EW<_A>{_nnPWtB@35&9my1JjmU+Bs9rmpfISaRGkuIO=1FJl!NQ}opEW>*3guL|rf3ne;WM;~>HlFi%UEg{+Hy>!6+{I}n70PYHnEE>5b~2ktr6KsT9~z<$!tT@fuj?`zJYEKBz=xVP>6os7+gyQJ-{e<xfl@aD3pwfJ4-+HG-1>d1C2smu6FP?(Wk@Y3k-YnCSl!f_B0Kt3+FwTqY7|g_xm84!<k`|INt%Yjp(?dh@znGBjd1c^?OAPWfD&1gwBk<OG~CYlVGLOIj0TTzC)tVhL1o1TFhF{7>RW|*~?hyk28u;sFgu2k!);KXljDUP;8mKlRYf(C($p26#2ZhShtn^tOgQdS`{;ux-?uD`QP$9Vydtq=VpEL%;?AB6kh9940ZUcMq}mi+Rb=SJ@4pKWPKY#2P1II{6IGdd9ge=L8>kiNhy5<qrRIl9V-vmUM7TL*obuqF+%i^%(^=UW)tc)Ag{%z#(ZclZ>Yu_He*57?PUi`(vaJrsEpD!<eAzh`WsrD3&ST=$Ye^;`dT`LgDLa65qU|OaJ=<M0f{{t?CroKN``@N`YLFPq~3xh%}@rC^EHIhOV2lQO$)cOYpSWx#Y#C@L0v(f=R;T_UrGx#m9mzKw<J!EdIha~Y&sS$dncQb5HCX=$2{29jqt_G>ErF|R`f9!P&yN0XzdLC*cpSdaQ(Yk3gN9MP@|{962#VSOr+6;rlex?Xs6E*X|zfj-|XOwrC7L!9n(yGq=Ylf2(baxB}^g9tO>Dvj%M4sJPjS2%IsJh4KLSi#L{!LW1`8^f$uAQ!zRgh63|Re&Jod^JU=5!R&XloQ0bcnf7`ruSbCcFi!6#lA4Lz8KjfZ6i79{3h(Yv*lZ(c+&opIcpfA;o_Rg$TSh$Cs0t?z+GW5<mEqv7;CvAm-tvP<<6Ikp)hc^l4Fk=!*+FX4QSf^zs!H?v&puaIQ9BMd~Nw}(eARDGb(I{&XGttW6v=h4$IDo<0+q-8((VFsQ(CHI1K6o6gw0<KOECPY$ZG)h<I>}<0U|SR1$jfk??j#tB=EZ)H!EkWY6g&NphZHmeiq4z-!Dr@R#iFm+gM155yI0tmGByjX%9H%L0P6=7IT6zHgp@BT4LaY>>6AB{TFiv4A)6{iwgNWeWr-vpLTkFw8;;N>qX6|)veyoaJE{Qs(>nVw(4=0F?5%0a6yQ3d8TwV5dM?J(?jZY$aWJm|UL6i+Qu&*9u+b!xEC7KfMp!QUluWKyF`Ik_GZA(HO4oVMDEKld7Dn1WbZUk2Q1vTj>GUFyQ>G(BH@+ewy80|_v<ccpGAU3LC1l)^8GwwnmM;{I%2|I`;gTkJY81wetUUs80$^>V24|dW+J=zH)dH8h+>@~3G;K2q`u+kRT8RXjT^BPNQCuQ7CPzaj#%J0BnAyu*T<*`g8xlP><oh~lvr$CpB>Pi<=ByKmuiwXN!%&2tijfiEk^2m(e#;~-bLiD+h4iEZtaazi=(q6Yj3qh6#w?TWLucB?vNnDQq9S<WK2)Vk+E{$J(<<op;%>@-$zYl0J9PT;^R!^SdYk;u2p*d(aHg%w&s&AfwTI0%0Wo7)>Lx@D3eKK{K8c@cskD1-Ew+IvGFiC%u~E3I_OQhyKwF<88X+<WR8MGR%TQho9<tF3DpxO<AUdXC-Xa`Wdw6LNgiEMN>-{5YppXrd){)cHmFlLFYaIaX+{vmyqS#lMF$jm29$uP*s>lt2A=(%Ooe-1BA)*l*nmUtkxBE*xgEfncA(nY_aBS(}f-%rrNBcmrC-HEgt>1Gy6wwjAXn6*r1=2?^>~i(2AUb0V4y`;~G6lt()>Ahf2OCY+KTP@q*-=3Uifj`)UbOU-%-nsLw*&`O9xm7c$5rmIX*&8D^g4(<<p~8#Nmy21cAAh94HH^y^Q;xPuj+88HP8VeYnsVZ&=zb{B6{(1IZT|(TEeno<l-BRO9Fn&iOraUyXy{H%|an}r;`RI!QA;&lBS`#BNJ??)zoOt^Mnur^N;PqE@g-H_CfJMLo%IZ>DT-a9cnVk$Qx%`w5<uaaHmaRU9sozxUXA?2NWJISPJ=hxeJRj*G52V>O-#4$}HoCXqrVox(~pRevg^<c}ww-vcpS@p)XL3*b$9D!Rhe4^^P5Vs`wsuo~&bDk*C}x!v1-S@tC^91<Rp}O&U7@puBdXp^7RL%0;eGgp23CT;$6r2&HBpX2x_pr1Egdgal7jj(%KSJsME78`N)5obLF2J`M^HYIIO`8W!J(CJwGWytE}gO%T7s9tugkkJfoiyiI1cd6%u7PhlOCmXMs+Fy#2ITa%6Dhr7&3CEwi@DnxXvY-?o-QLx)9ywi2)qaj(T&!Z?dk!LUSW@LNuVRKlattsrmc|vCWqp-s3J@-7C6=8RP#Y+~ky`M5zx$4~vN4M%#rKdGIdafEo7TqBzZ)7jl<_eERN5GC=G9Tv6#*yWR3--dp7p$WL2_kKM`V5mL?CIh5%q$JLz(P`02L)1}FmEr8Ek3+76h(jBad0Y%dU8O_O{6A5wy7wrZlklZ(O1y5USY2#0cR}5!PSRLnWi)$wMDIv{=H<aS;A7Nys?6Qoc|2@F@<S;k!S^p84Gc2_2FWo$xrA}DvHR3K0cVugV7r%2v2&lFfRkLDcgkf5R#ZN3`dn8E<_i&N=*pKFh>;B%M?$U{-umONRLK%*V(heC}Eepsd11l8-)H??Z6e!ynx$@^43k<ZYb^oya?20CPswb*p_U?%4W`hesyB9MJ{#Jn;HFhp4X>CDtjXwd4}74vvgoUaCy<oN+z%AAnB2lsMj4MKM2fB=-|;nD|{#UAVXtvyyn4J1z#{iniuH}URDyKAYx!mltRT-ySNP7ZvLd!GBM6Ymr4x*7X2*4lWm$pc-fl5s1}3+y8b$Ybn#ocwc;zwtTy}MnDSy!q0<oT2#!-*wfc>aUG-*B?-gEoI?{CYBF)Q=dZf-oKwP^D*E<YVk%^sV6S43oRK8Fsh>;GkBhQkUEl~`h=y}~C+6dQ;S&)ve9?{aJm_d`B)|Aj<C?sjYIi2xdkzvqAEb+q?DRTKcJa~Q(>wB7Z5?S}Czq5sP2WTTyS8Mg4{!X=sUS(a*NpGS0T#-h^NQPj_gROb{X&M%%(Dr4bs!YYU^GDD*Qi#GmFl~SpTUZU$y-skmGU-%M3Qm^wAy-SuT8F5kzNy^^roB8{E&rT}lNVXJt%7$J%hFM^lw;=ME9JmFE&XxAgNvWICrr!Lql1bZH*6jC38wWnn`z|^VO;3p^zCc5Us|=^y+JxpCVc`2bOQoULwiH;h42mohrpbP`D#-Tv2_#mngQu;B8KZBS^=>(fmhaO-2tMr;!VbupvkQxoPo1C(2P-uD#q!$4$)hM3)cfQM6pPEG5k`!8xSZ=KvhaHj2{E6vXqoPD@`0_e2f`<K1{aoGCJehA*-5#&P6kL?sQu9YNbK*I)M)`gyv%5+zoi%E?c+(oB>|ut>kwisbj8aKmZ-QQv`EP(mSqcfYrqK8)uMtch$m+xXfT4(MqT3kpfB7UluYEJ3WOdo+~8L+d!g@yy^YS!1Cio3s*mRN0`<jp6EKZSfb_NpTRWkbd2d(v68_%`bh($JRigKU7Ce=z;)=84zC+ig0(}6s(T0p7W3DHr-QSl>9+(lmBznO6_~;6`xFp2;C8nB2Ovi&6%8m|WhTZOuUqHbrYC?vgzk%QC0n6#2Dk53J>1}6XZjS2r>NJVnbCK3x~<jINUgun>e~$7jRDRW%OJ#`!SBO04=><&w#lv{69iT2D8hEMsq{Zik;2H((mH_;kZEtp^B{h|-r;&Mhk^rrFq{;9C>KVa#FJ0<wm_Sm)t8{LTpy0X)12q={4lx03%G8>zo@K-(WX_3F`3hlz%|iZ9(743gPAxbrz<DthQs|py~E{?d}S{8Ho3!B=5ue<PJA6s_h!As*Wq;^r<V9AxBCvY#Mj_=?=P142*(@TeR%GA;X^NpGNf1+K(t<5*6DJ|F{#{#BrVt7X3+G3^Ro{3Ofe~sxfQ0PtlsE!EuTa5!X;m<IO>lstNDg0>00A5BfJB7tVdup+9WAJM+2azV<mzm6~P|6RruZw(1xXTsKq)TThxE+g6{-HUtPD8%Savi?1`nQsK;{hs&kZ4j7@=ZL9$A&P^=5PLy{q4ea=iHR>s)+Vui$J)E)Kln4*g`O~~f1Tm3b{wHrXu^~;d-(xRIj)oSz<qVrkK4|R@0AretJW?zf0*!ww~!9L2cP4wqZw1{P5R}}7AUU`r<f_0|{R!1Hx0XG;nJG0fub$N)%gbdb_DZjHbr$r^JjTAO?1clqdm$IG|dZp+m$k=EX)*hjaaLr6Ufh2bm5<>ccNc9#Sg_bEb%1y-|&DF-ShLlb(L=K8`@XQ8PS-{_dQo+k)>UY;Ie=S^NK@!qL&ZSB;(=5^6VW2UZ6S~>6PCt`o2=#hs2u9{zHs3)HO10Of3KGKU)U6af>p=ErOMh_wdeI~IgzAPEU=09Zuy&mfP>p}r!uZ+(nJ7;qqM<|2kzcpq1Aw|wAmTjga}oJXjJyrjVM1@K)WK5?_#xix#)Rq|PH#+NuSe-R>DtK2beZG_G(&L;NR%WYG7ZAEdbFM*Org|ETe|_%TeH|(Kw2@XW2}~OYpBc3!Baek`h3!FpD&u6qLPhrTVE}<8!&xXBKtB(%hwm=waf&PAQ~b`TGJqIEP04v(^dqrykIem4$nD=J}i&@5}*Sub|h*bE{7gI1Oz;5VT?^)XBj*(s1MQgSR)-^Wgen;Ph?*K=<19}5v@-BTZ3qaK0PW*Z8Xq<(9MzBsr1qM8wmZGQrW8>y(3}^)uu-En?V)E8J{6GqDQq18rMYlOZ#EP=lBG%?@DIhgWRKIR~VqXMzfs2RM1@`x>~8XVUL;IB869Bla9unI1Ai&X0z7=c%Nz@+wPdv*FvpR7ojl0q=NA_&`#H|>jLV}FlLbB4Ftb8pS=;u3+`KPO&(oj08mJn1&#^G962cZ7-%OE;>nWcTmJUp8SM*zUbqug;3jOVrVpm5oAd|$JX{is8+1V+c7Rk=E}-u>5dH9^_NT}WizqsG@@eSRvDtn7jt(07X3?~b?Spvs1d}?HE=KV>gY1W;v@ak$_n<(Fqc|ba`KnEEe~9xVow?narf4HQ2OV2z`u3wGe+J?oozq_S%-80IZ_8@GK0kbGV*7PC;+r$uuf`KUF1h_6SNyo#_G5hU{gc~|amIW24B72v54|MYL38G9-Hr&KG^OK~^j>sHb?YZ2R9h@5K{EDKfUK^gtWeLF4gx~RQzPvC&%t`xns-(1So7dZ@LF72%;W}TY)9B2R)`IAl$VLG0<lDc&Rw__LX5(Wv-*VfRSQeJp)W+VmX_A_K$y$7(7E<u-8Q!3Y@_NL38!-F!K|9Dy;Y_XBMI-h(qZM26^3MD)Pazf$XayP&fKWeGh~u$r=*0TTX_Js!7K^qXkM;faSzE*u1(Nm$mNO&A(^QA4}p(O!suG(Ge<E87*p-^54I^(p(^aZm=OAMRvx8oKpVW=-VEOyGfBcn;{>3ZOgX%}z@TtIkXg1IrwemzQ-oA+C=U<}qYZ3Zkddo!H_D=C1)b}*uo`PqWloBT4A`d_%Ls`XTgWD=UaQUT;CnO%Rekb#yn9_4y1`}InLA?PLfv4>(GI1VlAkU7(Rt}5kK7YveVLLRn+EeVU-B8q!fkgh+S4=xKWe4Y@)1eaq6Hs=u$%J8_uy-uQZ=m^qFY>Bs93tOj0Z))*$n|xUyQL{z6os&=?1_amQa2Puky?x072o#h?0`=(?JfEDCFqorXkKKuPf1@!ML}dhu5Rh$(O*@E3)vW*;61GD}bvmJj<4YXtkzz9-JWXWx^AG>IS+Vl~BHbt?~}Ffa)7!H*gOr1|<!Yp#ICu)7HTN(qRTftR;<(z!_xSKcRdHSG8^OZB^lr0&2bZV$@<)7ox&({h?SHp>34I9S1Gl9JGF}gz}n4?+CR8u%M5t2v?}8KZ9DNj&m<#;F@L~+#jP}^q*kuT^Z#EzPO#S7h*(OtGK)<kmaFE+3CR0BO+>lWV9n<Mu83hT8wUC@8PN7ml$01Bl0nHo>e;+0p5mQ^&hCWv}71SuX9ASP|Jt*cOHa~P6b~8V|@u2PAaxA8c&0D%PBTjKt{irk0U7+>GMOFuOd=&3yhD>1-}GkC}wo!O0w`mQMvxYN}GSAClS`lLAeQmw5!Im)8qx*K;`3-!7tI7Pa#ao3ATR49Gocy9nplEXg@C|UQtjvf>VkX$7KA)bLjl=T=1f2zBbo)TQ2w!Z|X<k{2rIYeT?^eSPJ)X?(bnq+z0u;2PAPH;Q#{(4^QGQeCP#`Ex|3o*y1NsQDy!wB1hVxAc0vUb9sQz_w_ctD(kZ!eD20(4ew`!5BC|yE?n{cAiMCv7b|2jEPHhDGD(obL7CIMZ`Jysjhv-Kd42kKNq_akv_}iNB+(SNj6WL+OrtD8pUSvBf`AH1cDXS^jSuC5g*&+dwAI}xc#vNU&ci5*5IG=Wr3)In(R0BYVOJTB{5+_?OoLa~Ju^W`Y0{|Jg;~w(mfCsI`B-@bHUe!R!;Go}!#Hq3SqfBus{bndfng<CZ*J0pgZzN%m>V+Jpu*%AolUY>`wSGlrYjsEBaI)*txHeSMx@n#WtFQ5F-LEIJ|Ivid(}`A3D9G}chs2DqKYget7oEA1s$@Hg^K6cLX<^}46;OcSJv%dHP+@n>Ee4Q#I$84tB%3?Rw^=i3?hY!W`|KE!iqYiI%Po`r8a#jmn`@T#ejtY7UZ4XbEtiA8h6nncZAt!y=s*~34PxuiTh3pKefU0q#LPk%A{C<0lR(zv!5Z0yXdjILu`kbOt%7^j_7?bh-LUJg<^W7@0*wk12c3j6dUN61K6!u+zs#w)1XrnsyGi{9O7NMfzGPsi9%C9vk&QB5z5#ZDD(JwSQhsJy6Tfe$Z8A3JhBO(YnG?j1WdFmqHi{O5om5ns7PpM{RX-omc_k{t%6jajCC>!YUCgGDM90(&~@@d=u7Kv)?GmhW-;Qy8F1Y{k9!GP<3zNB^e$>TtQ<D0m?^vJsOKRFfe@xq4|d$(a06REQyzEKqxZww0#jDi&+))4kMaR)14NY0I%#CCA_6YtC7CAw1ZzKI7I)R-cgJ2h=v_vXM+npyUruswI}Da4$uOv|gccckUPl+Hjd9PT@4b24^(f2)9coRw@g7;paEc7EdYuyOCBMH(1rkq6=DPMR*pFsV`0zCDOEgw!4zWfLat!H!OoONzURcC&wSH?wD)87v>&a+-bSuw+@!?tAm!OPv7~Oig7+avRs<}9k9D!j}ht_3NC-IimWs(?0nR!e;E{ppTn=PTP^F*@189j(iLCZ;LHmDt~?V?2Z!~=4M)*E#Lnjf9SUG~gZ=J<Y&9PYB`{wQ4E%{kmx;`<(#!+n_Zdt46pQQq$XS=>jszf_%HW<@SYi|vZCOP_jaaAFA=kiZl%qfkpSvZ`{%+3Ik|LZtEbi6*()l*AyHPP%7FH_<rd`}1Pi(lzf7wM(CT32Yzp<jeR-yW+Be5!!rVo$1Ay9_^TLa`en1^@n6tFpst1=E?Jx4M6|N)Gs?)lZ>UMF*4-k;3ZdEG-jL+mvr&>+lZA3LN1pW^r)uhj6@!+{%}+zQ>C(BqJ0w;zy?5$_HQQA6nr6AI%c%!nMXZ>dW)-Tcd!*IQ$%go9DZXv8b3l`A<)TTLWNc~Qe=}T&6z~rpjoCMEqspzJC=k^P$C7KRtk?F3U*8Puof%@?ed-jUV+RILR3M1l*t^KBCSqP)Gz6D7FmDj;^|)*B`-+tea9U4Yz$S=m-D7k2aitz=k2A|)+3<u_A_4_P~KkgF)(@ixyOLy?dSejAacQ%d({BB-~_yKc)Y!XR|}5U(hZLQ$>q=94=jr(o@ZE+7D8qypC~5vKH=pv9Sfs(L;9Nc=|92B<qJOsGnYUA80=jB`~xsFngXyszhFsmU$)-KX7b<2kaUxYFOWf3+Q~poqs^E>&=ucvFC_H}6%q4+DKVK4+$k7kgTil}IpL8-;H37Qsq8`>YTd0fNV?)&?1rfNzmt}TP*H2ncM~u&-VbH~eQ^5EW+c|KW!?0?W&GkTNL{{_9iY_}X3J!O^)d<?uR>dtr}d?LM_aVCK@JF=TR@hi>O$2U(E1{s@CZy@@X$Ry&pK7az+9|>G4;#b>ol#P2ud$2V)mb)>4GI61EUKbd<=drc<=$hX=x{6VQH=CXzMf7G9ea9pLvV^%w8BQ@rF3Can8;o=aO%?pYxf_F2kK|B8*J62U=FqfP>Yz#xvb7^Tx*GnubjF3}!Al1UtH)VKSVI&SfZUbo1o@>+?idJt;HP(P&g*ArSBsS+RBlD;I2G5C5}FXdcR~m**Vws6M9RyuyJ}&}=Mc&siQn&CW>J#Tks`UOhgMx%8=*y8aa#+(yY(=p9ChLQbZF?^D;RIg8M7VC$bX{WIRIvpXsXQTT}bsJ^tLOz2myd4HH(`sDp_a_N(=6nYjC-!#3b2mva@MvAUF`~|URsFws9GFKi(bm-P_%J?ut8`HsdpzsR)IDJ85MDMg}qEuR8oC#|`-Zh+@S6PiIPhB6pJ6*lnir3w33_xIyt6^o)F9kz-n~$JB{tWcO(p+sQ%!HsJ1542)83p!k?M2;;lTgkhAB@hZV6R$_KHTS&Md*i@8)41wIF1#~3g-sYr{Hy~98K`a9Dn_`hbbMNqTD{Pb1&V(YMhkoh+9*gmH}~-aMIJ0VPh+i#EyO^8_9l=9S9%7^bCba46v>Zj&}?&|28n+lCpoc@<(Sdmp${fA>`-CU@m*^G3dB8gZav^@s9N6t3bvB)0dCI#RIaJ4?)GxmAYK_?EO%(c*v=`S2)(>gZt;|I5E)6Xaivikr-WbFz<P~3;f2FAA^>kGk3Y}`Fo=$@psVutA!F!4)xR!%We%+p)zMw`&DtPt}O9)^YGBkqvsB(%e5%VP-v(gjIYTTa0wK3K2Y{SzIq%(fs+0pYSCvE^BE9*K<4rVn8M~@ZC-3G5pWM)1jc*F{9ALBWiGR)eGQHF?$6QqRWv;?arqKe1ALW{BgU|JJVg?64Kgaynf|P3A|niMaZaU&G>@qd&si>f=)S%t^SrbkX1#~{yjl)tSFC4W6RNuIX%Y9ue}bZ)C1<(t!N=g{=A7krz-;iH2GphKDTN<d?k21?wUdLVoz%Rwlu<n&oajUIW(F|t$yl!TH$#{@%)b1e!nFTM$BGu&2#HBTyUcdL%pvXJjo!t1q&zZT`O@EvFs>N+p@STl?nMe$^f)FcEJn4BL@_O<Jt890Z}^)>q$^)SV<R~UsVuFp+KXI$m&jmtz0cbGIbE4(e~%6(Pp#fxLdX9Ph-^zu"""
P5_SECTION_B85 = r"""c-p<3Yja)2m4-j>`7Kks`hNMC0g_O~07H^V<&(+-23%zf!7$YP`+1%|vO^N7bktG>V_QBtN4r<|TJL(-t$%p&)2o+ne!6+})$#V_mp{Gw{>}0B%NMU--u(RC@ym<<efj$3n{R&l_U3<Hy!q<a|9<`bFaPk4Z(n`$tuKB3{m(zXdimqcci(^Y^6l~F|Ni>(tG7Shy#49LPd~pszIpkE7r*=d^~*1R<{58)c>Rj+y?puA+vBgt|K9xg=Ed>l`s8c=b@`juZr;B7=FN-OH(&qw;<CbDUcLG1`(NH3Kfe0vo0m5)UjOjzi<_^%fBjYXr&r=Xy^@Zfethxj&8s)x+`N79-4Cx>&kyzH{cAt+bN}u0dVF3#_~L^jr{j}9#x#y_hJ(NPGuION_HVxU<Hx`G<mUGu|L*rUzy0E)|N7HMpMLO{n?Ha2>4%^F`G_Heqw=8==E%8+mSf1J#iPfwv^-L6qx4ZnJ@}!$kC|%esmwl0JK8wsoW15&O9(j@ep1Y_&o*YBJ>-^;Tw=`YL9XoY{`x$!^nd?6tv~apzxm@4{!XIbeDcZ7r=NX#^Wg_etXFToJ_-w+>l>f_#~(iW;=e!o@b(=y|M}U6AN}EP0`(eeh@*y#Orw{hmUO1nX3njLQ1U!#&1dc{_tJTHXZ4qL@n3HU+3QS4bSL*k;3#g`!`ma_EB<UXjN`X|LWa+O|Cc{}`~j!)kH^8|XHI2|#+zBq!BS@*d6qe2th1D(u=kObrx<FAja}i3vyRe2O328!JPx+2mRUHW9NB^=od?UmsWIJ(!B>8?@r?(BnLVYFTFU{U(9x35P-{p%q9<d<fizvk$vLLj(-^b!@y1Izq|s6<CC;Was9h+b=hXPB^ruL}`fp-L<((jW$zSEcv*~%%f+#8V$x7={;)%Xw)>iu%z15?lYJ*=ayI>179x+Z#r4$b)j9$|^o8ALQ)n?=iPWDNP#`14cDC3<Vetr4)L-MF$DW#e;m3EYNrdk_{l`xcns#ShaS(1eilV_9WSw@Lrdi<kgg&CF4V{mBh07#ysXgux)OK<O_@M~py=xj#f;cdjyT8|xsLQ^w3mN}}1e52H(=d;h=fV+f2A<z+v0Eqd7oMsBVTO12{PEKp&5oUOrJiq&B33DrjIQF0*0B9_1DaO!h9Y<ozb3_WTDo=3vK)cS8u#4Eyp*|5LdoCazikSG$Sjf&N0fdOM4Sw3b9IgD5i{W&_wH~HUEP7zv=+z8`UC$?9#8T(L^l=)MH=jURj~-#H07gM>G>ygcX<kYGq=31^f#U%t3I=pwkAC)6koSK+B+VV}W!74*@h%84LLWWzD^2hnBLdSx#g2Oe+;pJ8LPmXS%x#1=4_2(ZoI}tC3NFrNbMbdMmWro)2#fLL>>AE_A!|Q+!_R|Av!+o2ful~2DrKMxrAamDS2|FI^6+|3wb$k5L!9b!?VML0V^@wleM)&4w2}y=;xDZj{3vXu)P{RVz!Dq555#srEjR<mkkT>B$*BNYm;mPx76xNp98Sc1hFGvE#@)^&*9V=+fG(j(H5S|<qy$RUz}kqzo(<H2hs_Z}ZT!YxW=|gXS)^rCK7chXHy<mg4K9Sm;I`H39yc-`1gwBgXLw7;p8=f5h-XYGG*&jS6<_J+m?aHN9giMRzUa@!S~7YB(S)BY%|d!sUsH=ocCo8j_c@V=_zEr$QwTWl*uYNC0Op(W$fyxCMspwegfL=C9F42(>r}*8cr+BO<XV>~Wma|u4j|FlHj>Bt4=i8~_PT?tEc?JvPF!2+z%AHQ7uP_uMlSkHa0;EtoLPk-18zNxfW-O_2%4k|zr*vx-f^(_fAk$!I*#>;U;k#2@%@9Ze)?UZ1V?!l)|brw%)mibRr>&DPS{$~>5A|NTXPmHc*2R6hx=O~5y2;b74wPPYrhNXQOB!eJ7XD66*a6AJeT?Mnm~xd8qH(L^F)ooA$)5t&|6s&@TxPO=m!yp`_|O(h66#n;INzW+ynU!LM6l@BLEnO&%wTWKUu4*(<XteR#_sUXlI$AXDurv!l{i(7)01`8$=j4VWmVLvZuHy#x~^7MS~*9PqZ^4`gF#{F~KSRP5gizVt*`r8Y{?pfKsegXl!Ciuj4khYlQhQBR-AG+tjaZ%qOZF?mg_SA<;33NfE6;>+okBF)v^M1u~)P!FxF9!b=VRn}|!tYzy*Z4@R!I2BP+~E66_LH<>>=RHHttAG3@2U^!b14RYNfpB1;nHV?RjJW=n3Zs8=<-f|(Q@K%5#7te(^woOqs5I;r309C7n6=c_wa4rND{}$dc7&Hyqr&;5F3+92)LYG*u5(X8(0vw=-(C}W$jsQZ+bK$+=!n+5*`{Tif-vRevN4yH|b2lO;90XX<qk~8Tbnh58L@}dF9UfWZTp<aZfMb8h?3+f5kuI>`TKJu`KJ}}EEbNHG1w0UfH6O9cN?LJc!Ps#HS@4@8w1>=_SB#Y_`ics%{GL}h&-W)R4W&xWpR^^;olu|WgBhbCi^U<Kq_`e=)0tW#P{}y38S;fD7lJQv-+WzO85Yb1%413?%hE*1l}Kal#GWATo)q^XeR(jm0_auDZW3{)LIj9Y1>teUrc{X?C6cTV%|YRMT48ZA0AtZ+%aG)fR5g`~06p`aoa3f)?E-y=x3`WU_?=!>n1I4p)g?m-&0;NBm;_TZbb%c>UjQlF$D~Sw2DJr{$z&i=bHY{N&o&in8|p#lf#0n`a^CfRIITKy#{4t!=jA0+L~J^op?BX^`bl^~JOJ(QFu~=vcsoZ_<b^oBu_ww7)Ef-qaiY$b;G+!KJ%&GA9Bh3;ZWGJHIhkzfCE^YegwAggR#n4#(4t@`h=Q$`xTjS!JsZ$N#TITp;?Z}&JZE=Z1_vHxune7SxO$TV6VMC83J2CG@Mrvci1<M^J1(nvUmh@zK-|6-ys_}RVE$jYZ+F2m`*C$Z0f8`Wm)nefc~p&~nV=(fG!JpW@<tU4R^(LKUxMf&o`s-)ie+W;8Xnc0xjDd2+_WLfJ>c9P0s?Ui&o*H{k%%rXhL<Kx7QfRm1al~7Hx*IL0tfsc=hc-SjcAw*LB<7~4JPoOWtzy`(uaLyHcW?Z6HR~$JUAKrW6sI+3AB7wwa#MxJ{l#p8j&OSlLw_m&3R*9Xa-J%MkG=6Mq7um56j39e0pna=2Q;0!O)>u&?-ITOCAzAr76)AIJ=dNop3G1T3>9J)FfxsN6Wd-AP?UWWCN6m$OC~UfJ}hONfJe$2^NmOPS8a>MioBr6wvdW=cOtMPPf4+Bw-j*wn!WwLjiF<1pywrd~xNuFc}YVdGys!z5|l$>{lVO<i>afb({uMQ<BHt;9xU>zG=4>KxEL`l<B}y-!pIa93$2($E{^QCm=um>37m(;O1I9hJ}Do5G$}Co1LqtnM$?;c06G*i}lMYA(=8{^cq~4Cze8dYvut(OBdb`;f^aSO^`muhBWul<@{isSbs2pq0s|W?2^3!1Bkoe@BFlK^m-cQHUK!m><!FgKCxIF4QJPR*9DQ`Mu0LtY5>$b3bBpIrb5gLgFmpa4o0!{DVR$uVnmyZK3h0KWrxK|q^N1-@R5pmU^z^ItY8z?lj2Cay2mTlJ|24rgQ#%MOVt670mT(&;rpgW*!gxE#k5#1sVrD>v;>RB)nG9|890QJVbT@FB(&ZTVgrj!W?&DHe4M!iWk}oxM>)XV%U(k62xp-FR^bD$^n6;t*Bag-@r0Tn@RU}E)kN}`#bcGvmj)@}QOrD#I}4oY%K9b-ScFufjEKsksiwRwptD2PWLiF_+|MKl;Y5Fr#HH(hZ&M%}i+)Zd-g*9ASp2Vi{@v&qOS!(#4(Jn?!<mZBK?g_0Nm}TEB25dS*s?V^p|MzP&%S4+e<1BL^aVcwl0V?}=F;JgdG9#y!_fzmC(!e0>ktfXWchiA#^!AJ1wQh=6-J#6b^sa2ZGe_XYz>zI(8h_av-*WWR>7L#9yU%uzo!@XKxuq1OODSacEC9o%RxeEJmulmK;eRQwA*}&l^1i23cUcg8R2IcQCk5!KE=+Q(e(JZxGh1psO@aVcI-QJHXSY6_*XOk0~5Fc+`wtFL9hoqh(mby*{lw5dO@$IgU3M|nW$d@^?UU4wae!MR9gfd`|P{&4FtPar5o__b4pr?oF(-WZN{}uAQgpRGbFb)z+_Wt4X64ZQrWZ_rwO!+-`QCBbAsy5^Pdw`cb<P=lHh8FZ4R6R$XI11LPc{|1HgqC%mq9rt`QT5a(5sLPA51);o#OBc3LT=VAKs@&1|r4U2o?w?}OC-5Tvr&QccmqFQGknm3oH0vM^?BlX8RzHHDjE;}!2-F4Qj)M40EOgcz4Ju7Noet1Q}G@}64UM^7Jg4wzCA29sS&JRI%@#$zjx<yBloHq$v15iDxh$Ou#uPv9JjX^(<1EA4@Gbv}a|EKu*VCCE+=7k>n;T5fJ(YUu!NkJDxtn|z6zi4tnbGnqBBzT3=E>f0@`+L5q2x(Xau8PkfJAGLnJd}`}>mLnRqpdLnTy_|WXS;Q`b@8P+#rD7HsQ7*%{V@$Sv@BwrvhSpcG)6J>XxvJo(n5`<|k!^~%;ykHuJhu_e#bhhg(@c7{Ue?trn{~l%WIZkJjwwG*@mKvGghoW*<6Q}GUxv-bama+Er}di6heZ>yt0MwL8a6)I>QQ_J@rKNOER#<}nTaqB(q#S85SkxT4LyVXb?GUxZ1#=ABPQ}di+W%E-!_Sx@3e~yJQ}7N&TNsp_SI)&{xjxtU>PekDYnY4qI}deA-`r42v(jT%x>GbMX<vE`1Uc_T1$VJExgKJ#BA3AMt`nyEAOjz<FL@wPENG@uWFW1X{P-zu-O8smnuF*&9>XdbtTot4ok4zhSv#wfx8U5rt>PA#eU*ynz?d(XISmLi0IdCnjopG#SFL9VYZx2vU_1XN!PYp#{th4veiDF!_qB6-p-0~blIYFPrQ;thsopfQoD4s5K<Ie53<;=xvGM&^<X2+rC9m|X*L?i?LU!|*eArg!LPHfxdSVdlJ1&tKHa>|uw5X|)waVC46TLF#Y+lpWdxmb$YbAN?rIv(k7ydVe$=LQB#^aEHvDvgLL@ANf!53883{dBPq~8!!L9>N({n-@v9{G!8j>C6*^*xt)()&{KBkTFR+_s)fCUFvb&czV#9>xr;Lz-{!#v<oeHiHSO4GhUU)5+a#P_a+jjM36wp`2R@^|NvSlqPj435F#qIp=X!qO4G2L93=5)o!mHOF9rEwLO{D3<=)Al{o7Hy=|NChFmz*)->-AiF-r=`_Gr*b1D$NVfa3w@@#57ZE|XG>GEEO^C+Lmat^oo2TBx<VsngINnma813#)t?$aQ?M~Vnh|pCaw!PHVkOVu9O2;n_yOp!pjuCAfG<L!Gj8p6^w-Yo7w&6_4=0ls{TgfAA%kl(gGS#9s^N4I|wt`XZNlI}B>_t}x$G3yQJcQG1h8!tdcjkqxC#sjggmVF_3h)QF1j=^xYuk^o(_=Mq0>9D5q)>p@)m0FalNae?8OSC~>(#NxYWETgu1>~{vT8AG!Ylw<wIO@VyerFw7ine#`CY9nwbg({RLh#xbquEZ3W^jRoeVPldZAvhP8cNKugy#DE%{BSUC=Uo(3T9(**+d>c|<qk?bGjyD8_XG#enXGYWG<makVWECpW89V>si%D~92@IBYhoe&@e|e#)tsSi6A!-s11<AIF<qo&3noFUay_un?nq;)ig$$<be!c9UeJYL_`g1N<Qjx8I|wx;7_3ZZ8eXedTH$n`pDzK)xl=F7~sTcI5(%4mY{We3H{?&d9O%d39b#nP4zglB<Jq#(zdYSaVz+ZfUheX^w>-;gHCtAlu+?ozO>@*@@W9Z1h#RwXe>hOwn?l`mn-E?c9@9XUG$}L=1%(PDhHCP(aiC6?oPCZK~K4kUz&_I=>kC=*MOk=IuirYh;jDrZJGj4GLK8Pfcrd+v;vgc4IrR3AGFMAXR(%_eFUM@Y^msZ4YDV2X5?iUYihL4vVRMWzy;x7AP=tfHSXif)+Ysf_On+U@inztqFes-$L+BQ8sYjp5UyV!-Wja(3aC4NR~L9h%LAN%-GEq6#}GCR*4(CPOHiUPK-;2FxY@&APxeQEn|PqMsgHtJ}NPGUn9BMovY@Cs=Z2pK8&TTIgfcdEMTOC+jRI6kaT|6-luly(7nZd&J1@kZ)ZfemVRd|`ApZRycNKSp9(E=A^j|N@KyK~SP>mcfdYiVn^@^8?x$qP){>cGK6(ND#JQ(67}!zzxhKutnBN{u6x&0Fk_sE<>Q;YTvAj}kUThIhHKvV-HjM-Q4jG}(*~!d6n1Y_$XL9Le7|vSTrj}Ncbw`eSARh<+5abt(kN=(hkq)!RK#mg#$*O4dgs^kRCgHv^jn?Y|N|?p~`<FM+d8O66DWS9lHg0Oy6A&N2xthk)k_n|rD#?`iI$exxCXR@fr`^`X$MJx58Y!b^j`Pojk_MlkX>Jv~T1|3WinWjTL?OL6?44Go1*aqb#Sei0)#*ZGG6#&Ie}vD)z7yRW2N<2*k{Mp|UrrqjgCEYsz->p69hN_d0kaa=|8CQz)8O3#E{Z>FZ?!Vrow;k?E8=t!@2ZYAB*%Kw`>Ypy7<!d#luoMi<7IfpUh;l@%jCO9-<8-&SLSu>kul%qr5`2=(QvN_Xx#~cHFd~ApL5NI4KMr{_16@C-WEaP-pZe|xV!QAJDbVJ)hWKVh=iU2QtOp27U@7;vvjuI_ch7CY8q^yL;|`!h&QHt3DEKBHH{Ibg%?98TU&V3hAi8J9Bcv!B95N$jerPyKJA4!R*tZ1(_z!AEU+`1=LjQsQ(?y@A2V+QaI_0LM_U_hk^sbw4mUJnADoSaoO&@IQ@nQs2tyE;2@`P8D(M`<YExKKtF{^1{vl<_ab*rnkR==ph&a4tr6J;$YHee)>2AE-P!X1Gu=&;g;v7~Ye6C_Jr(qyERwdb)>eG3i>il1+RxrGe#QT;sSr}XNrjN&qv2*>4jWXTBFD8-WZGI?n0QLfv9NLO6a*>O{3AzL6tDf%#SJMiZC6`MuQ5-6^B4ZNs(GomoKY6*jfAZav?|@}E-BqYeI8|q874F%Z13vBr+&ehzB(#szPe)%1_L*$Tc;9g|doxZemoo0-SN)uj{P?He$(8H%<2@@a6};{x+tFqHUIN`=26D3CWpEZQBy0`_65vdG3L%zaNB8Blz%1m{WuzUt=Y?h)!rVug<Aa4T)8Npo=(QJfvSgfw*M>ng?U0-+v4DZ`-P+cu95mB4J6keM>nwKW@QI`8*$GDD*bQ}fLY39=o@{li;U6u-vkQ0+nQ`g14DtlnMu@g9H*<__7-~z8Z837vcdYz~1PzRH;KyR+j$-X|Wh0I3{E-PbIlp(RR0m^PSe)O(0JdeMB^O;fi#@Y<Y^+G7=X^O0RLb^ba}r}ihMlB{p`)$%b=o-uwQXm%nLm|ap>9=Q^65=0y&3U)9}=+ZooE&`@PSUSTq@l-6L80S$B^lf>wnIM^0@BUdqkVi-3{ev+I18;cro1SI))&Cw2eJ9RWzN&_5LWUw#$_{w1t(v?<k+c^z*!(w%l3wv$j8OJpWnSA2*(VPfs}!>Xk0RyW<=ko|viu_lj)2iP2Pw*C06#lH=*LiWiA`p%`>z6dRIkr1D+_+Y2pfagqkbUV{>P-#gHKK)u{571m@3;TZ~?Os)(aly{Qo+<H7)w6N;^n#**7E}7Nn%}%M<*8Pn++R~kZ*Db)`>z;0GV=6pI*iSZ_;)3~vxt=x7Y7<*tsTE8JIhwgQ=j!?jI-CrUliQ5EKo!u!Pqt!Bq3_ADLvka`boO=8pN*zAJB7Nunk$rN$lHe(EfF1NF@V$YVm-S*OE=)otAd@g_c1ku@Y=e^E7h}hmdE<&XWx~57+2m`;`OiIJ%v_6-HvGsiA@^1p^UCBXicF)=DfVOg<s{FLohqFx{5bf{;b%#@%(4S){W=imom5(TD__zdPkvTyY+SB9}sIhoR|1Euam8oDO2<oB}cuhw;lHC{fy@6#N<_6&2@majuX@(z1=y@dx5n+IQQrUE5l0(YZf1Kisx)5{@3MC1Z|+y=|wEWUPQ3Ldt>vA-U7Pv*XfdHhwNG*o8Hpw*yNs8JfVu$%+SqMgaOmDBmdKs46Hki9xVXjMy!TsM>dD$xp_8w;Y(fMW&R_tYsso@sJ4Ra9mft?b<oIbqHSuDX;)k3P#mac4=^xXK*UiDTM)qA>SnLnF#obckcH2%0tSmB>sCs8rTVo3{6B)PX0-"""
P5_PATIENT_B85 = r"""c-p0zZF5w|5rsc1KZcr~x1RY(02`_Z5U}I&r!B>nK~*3JOT@{)&vSZKyBi9OudtycMt5(&p3|pC|2n_9ytumAUH*J_{nPo|i`|EJXTP8S`{M1z)$5x#yMLZv{e1fGx4-`W(GzbjU%%1O)qmfAxV*mEUEiGFe7HV)zq`Kp>E`m+tKG%>59e=p?=H@-P7dE){@=gO|8w@+)%ouF^7R$Z{_@-TI{(Bg_D9!$dC|^FKC_hTE~h%;yL;(O-}B|0@1H*1y?FfNt0xccKi&QC<k`dLKb)m|!*a1wGS9lV);RK(lINUTE~%HUhrG5{tF_$g@MauY=`APo(JVC|eVpZ62lCkE*7a4^TmQWAZeC`oK8!bY8yq!Wg{khX@ais8z4|P<j?ue3lS2w^ddsu)F;mLzU|rGfYMT8vZhij;;g?F7ZT4(!m?`|hTXDi_?`~cr{xW+{#oc?Yb%Y0mm-5V+Hw@lzbfEQJv%C+S@BSeCIn%V-=9r~8pV1MSuetY<h&d}Gp3)>MjXZg^DdQZ})JNseDV5v7I`(!-&ff681wLcnhtEk(CoYukIw9#5(>ePbV@5iP*UBGU8l%=0wf9o>A?lX9on-$$cAh<dwtM(MUY=cE{Srsd|N83D%YQt2xc}MiZ_gh-dL?t}snN3jlEfY!ka~4inKQRJqaUp$8!f3FX~i>BU+pb^`zVUNud)n&uF@^nmT;L+Mp~g+Yr~C?raF5mI-(R-M=P0j<EI=EVV@a<sMY5z2h&f#vnWv7xB83wFTZ<&$;VH=eZ2eT<)goU|LEC+ACDfCS;<wKBz(3~<U#RK=`o*a49{wI0vR*|ky<J>`roK>NnY{50pPOF*R4pWWPNlRhR#Yo`Z{H+1$=qOr)aG)J3sG0z;HL5G&M-7puAC>+h&sk8<1WBEF~mXn{!kh<xY(oD5jqv03Fc?(`%W;n@4n{C8rmFKoKzd!S+_1lTYiprAB5`6dxedXCn{JsYw*y%1G4+oyWaItEXBg3|HYJ0CCLz2Aq2yUh?ddSVm1Xj#>w}rFUz%6*$MFDkGLnLBy7*4Cx>N8RVB^^qXKlawa!W*R0v#)j&SRTvT3Rk1``e)S&eS)~zo#unxRw0?%|L?bi<WH{u)*=y+l&Hhsl5vW<iXw6^oZNPnh=Q{bd6f#wnCSp4%iA9O3|g+2(S)d;Ol8XZ0<qiR&8dy}9?Rb9+dRHUdCH0T&KaBVqQ-U{<JqO01(JkhGJ8~r@R_5_=lHaeOHgVPz=@=R6SpGZz~x)ENlLk=z<2zx8kMQdGS-Mida$puj)+L*NTd_WzlYw!#dz{H7)j{-pz6F)$K`RKqWp#G&q^s3QLOhfFFRdo>GAy!0$hDu+<9X<r^vrXtZ^3WStPcQ+zjLJATu@&>6P5tZ|=*pcsAr*C7ZOreSe1X1d+DKWHq0&H!Ohq^`LxRS$M+dfGK4(A!?ot|>8ahI25#-3t(KjKrm@9>1gF|LROd$HP5|`7#RN)x>M#z&xn^0}KeXq}GH!4i%wi*EHfq4ZDhi_{@e$cAe9mJjwWRLx=kgw`*WnRqXk!ct-HOBiu@U8eqEk*X4e!v7eLLQ5M9`eXHMBrdP@|HEni3`$_ZfT_r9YF!mtt$!en!14wPm+B}?R1#znA`?={!6!R>vNmuazcT_7+?h8#dLtfI$8`x4kPuz?=;adb;=DQk4~8qMXn@EK-q~0cR(vz3!P=qmW_&;psl1%O%OHZig`$5KpBh*hbBf@v?Y8CGsrZA#L<mA;FSp&n$q_}n#Q0HrT>s<zJ@fz$tzO?;#{H)w=MK4&4!*h^<aw_M`I*$GUAK;>u69JIt1ttZe3#BNdI(ni?mw~z>)`a4YRYevD{8cncs%l6bc96)&fI?92X}tw4CB20b)!?_4z}fkXm5YnHI=KxSKSngANXC2Q8zIk<YBCD>~%A_CfMbm#ywp2iSy4AWTG*9YHIT8&HcPWF$6}SS<n3p*exfr?&KgS)ev)zpN7B<X2f?Wn;J@D3L30NjZu4D}Xz<ug)+$vbc)NvM*Vc2qdW^Kyr+JhFwO_F^*~Nx)ut$3K5F#Ce7L|PCVEQ+>c|}iaChSDeNQ&)K3(F0QkSN^0M&F6$BMv-5X-`9AeF!)pdeAwv`0-+ldRG0Noe00xFcT&^tqS_P~CZWW^}Ui0RvSN}y+|HYQgMh(Jv!1smuaZLX(&Y}|#rcCyA>+4fWHogjfZiZKdW<JN<6b*RVa`-&c4iD(&}gJ?m@7hO*r9N3D!Ce_PkpHKuTWGa9p=}=$Rd`Jr;s739<_#*jmg1)(79iC!O)0X4fjDuJ`wEmO;K^N<x!hpdpz(cbH0ixGcBG71Vp2yMj-LV%PzZH8BBYf0C<b?kcYy+`nv8?1!94<H(32e%f#b+9^kh(IL396iU@KxB;C%0m+g&b!kt$0%x_Nt~Jl4X+OvdTiaO%q{AD6YW|YcMNK4<;Tsap5l5!-1r02o3s0#B>JCP^4kOcU<~($hqsPZD!U66F?|bkyle!Xr0Aj939yPKKdEW(?unHtp#XM=|@*S*3F163F8Ka3TBU+A?>imc*=thtR3xd#a@R5*^RYrBRFF6f#!xApaWHtS36DRK)^wAo-C-cxCMIVqE>R(`WmUBCP1CGpc+$Mli$^(jFYzLuS{+2aB<wlvdSEd-yM7udMo%)KP>3I^uoUCaYuxMy5L}!liI%JO5>j?{I@6}{gN3oP*2H#wU-kYz6yPM!{*5c*O{)Gs_r5}TR~`S!ZbY_Aqrz@Qe#PQH8{(32?n7m6(A6)w8}{k+tK%N5TAz47ZJ?x>dfP;dzKspkzP#@HIu1ki-8(onbgT4lG59{#g<lK(j#H*7C&-i^Xx(gp+SLWSN7IkqpL;~X{Xoo+RAi|iQw}XY$q4Fqi(@Z5-&knNhCG)OZ?FFNymCw_+gnRD>NL1DzT|l{iXv5!zo>NettB5cl@Z{Tk(UoM&S2JAFbWobn%4LpqnU<uVPUd?L(7&D+ZjdBQy&b7;qDGeB!`Yp(k2)D|!%!6c%u^V%K%SfHFhD(S<0Ab(t=b(7jhmbzS?Z8k%M5ky36inD&_Y3hd~Ucf$@U#?a-+aA~}e*Uhx9r2z&6hX`KF84|t%dxqZ{QE_KL5{B-|=HZ%1o@`#brhdV|=tHQ0#t;HftZr;|JKf_qPHYl1?er0{xb`D@WOBpWg;XYJR1$h~sSX$A;e>S2Z0oXwPMZK#yuL?|a>70!T5<h`>S+9q==mSXl3ZT"""
HEST_B85 = r"""c-qaE-EJC55QVR2rF{opz*g69cR5$YMp;XQMTSVZh=hnu5|kLc2Iu*=x*Ebf!raW18ZPFWQ*}<waeesN9bPt%+vm-$d)_~7es$kp*DpUtyVHGsd)xgtdJpx2^+NT+^&;J2|MvR0`TMW?yW4I#nSD0v9yW*B<|wJdlohU1m##S#U^1_ivjG>cC~IaVuP~>Qtezn!%q7lJqJrAp-F&&*Kfd|19S>g?%k|sS_UQL<y|2d8>1y(MK6$wRIGHV150k~G>2k4}Pybym?`I!YAIFR3<Y9#g^(vLCWY^D6qnn%2Ve@$0?sp@5H~MF?SdPy4qi@^ao2#$u-P7oG^Rj<Ej*jcEzxs4?q03oQzzH~tX@x_|B|upG`4a?Y1_7xRGZYBSf@rx62yS-!{yvu;9VcjXEFx@`M8r_J9wDok1yT_%feFIclw*X<GB`NuHKwVYq!Uo6PptqYX%(@ml$&5$7_><T39<Dm*Tx(O-F!U0i^c@d{@k6kBeY_gD=7K+<ysMmXh?=?){v2nEumMCiHg6Fq2Y15;E|qN(?nC43YnIeKsTD&o0T(V;LI^?$?CLu1x1O5M~6M}z@&Mm3qi=aadHkB$)H$F9*Pg4)iZQym792wg<&9g(@(PxC-I;s9x68R+&NTRBoHO6N>#^C;Wg;eLi8(&QHhuUEB;DsaCmO%J@uHIvn2PChyowBqzr+~-lokh)lxJk0V^~`k%%36V2Nkl{Kxj^es}d7`++Wr5bDIaReCE%ET~I{8kq-<N3DrXVlv%x0cm)7exbVrx}k8`VTMvPvcOZ2B&LkrOfWKo#f0Kl|F>e10S5X7?~>>R)EAK2nhA}<HdQi7NM~-Pk#vM|dJ9;xF%B@$d!2bsbfxT%UE48>Zy_be8DaRptL2#xz&U9uQTcDa#tbpg`-68*bQ4L?wDT~twvM@urLAM=eHazy%HP&8$LAp?`i1XLqzjpcHHU2`q1aOx9a*t$&<YWWK6bvF1)sW^W*cIrU-<rnx(}gBd<c1$mJ&*GU94G}YNkew4@szPFP?gNh@sxo@0{vF$-fN3I=yJy85@zduro#9L_<lD65f)Pu#_Q&dSAYCs=I?IRV&H<r3Zv@mcf_J5Z9X$qdhYgKCPj}`G=AW{{sYiYe@"""


def decode_tsv(payload: str) -> pd.DataFrame:
    raw = zlib.decompress(base64.b85decode(payload.encode("ascii")))
    return pd.read_csv(io.BytesIO(raw), sep="\t")


CRC = decode_tsv(CRC_B85)
P5_SECTION = decode_tsv(P5_SECTION_B85)
P5_PATIENT = decode_tsv(P5_PATIENT_B85)
HEST = decode_tsv(HEST_B85)

assert len(CRC) == 768
assert CRC[["sample", "scale_um", "model", "wavelength_band"]].drop_duplicates().shape[0] == 768
assert len(P5_SECTION) == 128
assert P5_SECTION[["section", "scale_um", "wavelength_band"]].drop_duplicates().shape[0] == 128
assert len(P5_PATIENT) == 64
assert P5_PATIENT[["patient_id", "scale_um", "wavelength_band"]].drop_duplicates().shape[0] == 64
assert len(HEST) == 18
assert HEST[["task", "specimen", "model"]].drop_duplicates().shape[0] == 18


OUT = Path(__file__).resolve().parent
MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.4,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

INK = "#252626"
MID = "#6d6c68"
GRID = "#dedbd2"
PAPER = "#FFFFFF"
NA = "#f1f0eb"
PETROL = "#176d70"
OPEN = "#6f706d"
TISSUE = {"IDC": "#477a9c", "PAAD": "#4f936f", "LUNG": "#c06c42"}
QMAP = LinearSegmentedColormap.from_list(
    "nm_warm",
    ["#fff7c6", "#f2cf68", "#e69645", "#c8543f", "#842d58"],
)
QNORM = Normalize(0.20, 0.95)

SCALES = [32, 64, 96, 128, 192, 256, 384, 512]
BANDS = ["LT128", "128_256", "256_512", "512_1024", "1024_2048", "2048_4096", "GE4096", "ALL_NON_DC"]
BAND_LABELS = ["<128", "128–256", "256–512", "0.5–1k", "1–2k", "2–4k", "≥4k", "All"]
SAMPLE_ORDER = ["P1CRC", "P2CRC", "P3NAT"]
SAMPLE_LABELS = {"P1CRC": "P1 · CRC", "P2CRC": "P2 · CRC", "P3NAT": "P3 · NAT"}
MODEL_ORDER = ["COMPOSITION_RIDGE", "FROZEN_ENVELOPE", "IMAGE_ONLY_PHIKON_RIDGE", "MORPHOLOGY_RIDGE"]
MODEL_LABELS = {
    "COMPOSITION_RIDGE": "Cell composition",
    "FROZEN_ENVELOPE": "Frozen ensemble",
    "IMAGE_ONLY_PHIKON_RIDGE": "Image only",
    "MORPHOLOGY_RIDGE": "Morphology",
}


def matrix_for(data: pd.DataFrame, value: str) -> np.ndarray:
    lookup = {(str(r.wavelength_band), int(r.scale_um)): getattr(r, value) for r in data.itertuples()}
    return np.array([[lookup.get((band, scale), np.nan) for scale in SCALES] for band in BANDS], dtype=float)


def style_field_axis(ax, *, show_y=False, show_x=False):
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_xticks([0, 1, 3, 5, 7])
    ax.set_xticklabels(["32", "64", "128", "256", "512"] if show_x else [])
    ax.set_yticks(np.arange(8))
    ax.set_yticklabels(BAND_LABELS if show_y else [])
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_color("#5f605d")
        spine.set_linewidth(0.55)
    ax.axhline(6.5, color=INK, lw=0.55, zorder=8)


def draw_response_field(ax, data: pd.DataFrame, value: str, *, show_y=False, show_x=False, contours=True):
    z = matrix_for(data, value)
    ax.imshow(z, cmap=QMAP, norm=QNORM, origin="upper", interpolation="nearest", aspect="auto", zorder=1)
    for x in np.arange(-0.5, 8.5, 1):
        ax.axvline(x, color="white", lw=0.27, alpha=0.75, zorder=2)
    for y in np.arange(-0.5, 8.5, 1):
        ax.axhline(y, color="white", lw=0.27, alpha=0.75, zorder=2)
    for iy, ix in np.argwhere(~np.isfinite(z)):
        ax.add_patch(Rectangle((ix - 0.5, iy - 0.5), 1, 1, facecolor=NA,
                               edgecolor="#b7b5af", linewidth=0.25, hatch="///", zorder=4))
    if contours:
        physical = np.ma.masked_invalid(z[:7])
        finite = physical.compressed()
        if finite.size:
            for level, color in [(0.60, "#4b4d4b"), (0.80, "white")]:
                if finite.min() < level < finite.max():
                    ax.contour(np.arange(8), np.arange(7), physical, levels=[level],
                               colors=[color], linewidths=0.48, linestyles="--", zorder=6)
    style_field_axis(ax, show_y=show_y, show_x=show_x)


def draw_bubble_field(ax, data: pd.DataFrame, value: str, *, show_y=False, show_x=False):
    z = matrix_for(data, value)
    ax.set_facecolor(PAPER)
    for x in range(8):
        ax.axvline(x, color=GRID, lw=0.35, zorder=0)
    for y in range(8):
        ax.axhline(y, color=GRID, lw=0.35, zorder=0)
    for iy in range(8):
        for ix in range(8):
            val = z[iy, ix]
            if np.isfinite(val):
                scaled = np.clip((val - 0.20) / 0.75, 0, 1)
                ax.scatter(ix, iy, s=10 + 34 * scaled, c=[QMAP(QNORM(val))],
                           edgecolors=INK, linewidths=0.28, zorder=3)
            else:
                ax.add_patch(Rectangle((ix - 0.27, iy - 0.27), 0.54, 0.54,
                                       facecolor=NA, edgecolor="#a7a59f", linewidth=0.3,
                                       hatch="///", zorder=2))
    style_field_axis(ax, show_y=show_y, show_x=show_x)


def panel_letter(fig, letter, x, y):
    fig.text(x, y, letter, fontsize=9.0, fontweight="bold", color=INK, va="top", ha="left")


fig = plt.figure(figsize=(180 * MM, 250 * MM), facecolor="white")

# a | Discovery cohort: 4 model rows × 3 held-out patients.
panel_letter(fig, "a", 0.018, 0.982)
fig.text(0.050, 0.975, "Discovery cohort", fontsize=8.1, fontweight="bold", color=INK, va="top")
left, right = 0.205, 0.925
top, bottom = 0.935, 0.515
col_gap, row_gap = 0.018, 0.010
col_w = (right - left - 2 * col_gap) / 3
row_h = (top - bottom - 3 * row_gap) / 4

for col, sample in enumerate(SAMPLE_ORDER):
    x0 = left + col * (col_w + col_gap)
    fig.text(x0 + col_w / 2, top + 0.014, SAMPLE_LABELS[sample], ha="center", va="bottom",
             fontsize=7.4, color=INK)
    for row, model in enumerate(MODEL_ORDER):
        y0 = top - (row + 1) * row_h - row * row_gap
        ax = fig.add_axes([x0, y0, col_w, row_h])
        sub = CRC[(CRC["sample"] == sample) & (CRC["model"] == model)]
        draw_response_field(ax, sub, "q", show_y=(col == 0), show_x=(row == 3))
        if col == 0:
            fig.text(0.132, y0 + row_h / 2, MODEL_LABELS[model], ha="right", va="center",
                     fontsize=7.0, color=INK)

fig.text((left + right) / 2, bottom - 0.030, "Output size (µm)", ha="center", va="top", fontsize=7.2, color=INK)
fig.text(0.008, (bottom + top) / 2, "Wavelength (µm)", rotation=90,
         ha="center", va="center", fontsize=7.2, color=INK)

cax = fig.add_axes([0.944, bottom + 0.010, 0.012, top - bottom - 0.010])
cb = mpl.colorbar.ColorbarBase(cax, cmap=QMAP, norm=QNORM, orientation="vertical", ticks=[0.2, 0.6, 0.95])
cb.ax.set_yticklabels(["0.2", "0.6", "0.95"])
cb.ax.tick_params(length=2, width=0.5, pad=1.5)
cb.ax.yaxis.set_label_position("left")
cb.ax.yaxis.set_ticks_position("right")
cb.set_label("Noise-corrected correlation, q", labelpad=1.0)

# b | P5 hierarchy: two sections nested within one held-out patient.
panel_letter(fig, "b", 0.018, 0.480)
fig.text(0.050, 0.473, "P5", fontsize=8.1, fontweight="bold", color=INK, va="top")
b_left, b_right = 0.165, 0.925
b_bottom, b_top = 0.300, 0.435
b_gap = 0.030
b_w = (b_right - b_left - 2 * b_gap) / 3
section_names = ["P5CRC", "P5NAT"]
for col, section in enumerate(section_names):
    x0 = b_left + col * (b_w + b_gap)
    ax = fig.add_axes([x0, b_bottom, b_w, b_top - b_bottom])
    draw_bubble_field(ax, P5_SECTION[P5_SECTION["section"] == section], "q",
                      show_y=(col == 0), show_x=True)
    fig.text(x0 + b_w / 2, b_top + 0.003, section, ha="center", va="bottom", fontsize=7.3, color=INK)

x0 = b_left + 2 * (b_w + b_gap)
ax = fig.add_axes([x0, b_bottom, b_w, b_top - b_bottom])
draw_bubble_field(ax, P5_PATIENT, "q_section_equal_mean", show_y=False, show_x=True)
fig.text(x0 + b_w / 2, b_top + 0.003, "Patient mean", ha="center", va="bottom", fontsize=7.3, color=INK)

br_y = b_top + 0.020
fig.add_artist(Line2D([b_left, b_left, b_left + 2 * b_w + b_gap, b_left + 2 * b_w + b_gap],
                      [br_y - 0.005, br_y, br_y, br_y - 0.005], transform=fig.transFigure,
                      color=MID, lw=0.55))
fig.text(b_left + (2 * b_w + b_gap) / 2, br_y + 0.002, "sections", ha="center", va="bottom",
         fontsize=7.0, color=MID)
fig.text((b_left + b_right) / 2, b_bottom - 0.030, "Output size (µm)", ha="center", va="top", fontsize=7.2, color=INK)

# c | HEST specimen pairs, compressed from the paired-specimen mother grammar.
panel_letter(fig, "c", 0.018, 0.248)
fig.text(0.050, 0.241, "HEST specimens", fontsize=8.1, fontweight="bold", color=INK, va="top")
cax = fig.add_axes([0.105, 0.052, 0.850, 0.160])
cax.set_xlim(-1.42, 6.0)
cax.set_ylim(8.7, -1.0)
cax.axis("off")

metric_cols = ["q B1", "q B2", "q B3", "q B4", "q B5", "graph PCC"]
metric_labels = ["Broadest", "Broad", "Mid", "Fine", "Finest", "Pearson r"]
for j, label in enumerate(metric_labels):
    cax.text(j + 0.5, -0.62, label, ha="center", va="bottom", fontsize=7.0, color=INK)
    cax.plot([j + 0.04, j + 0.96], [-0.18, -0.18], color="#a7a59f", lw=0.45)
    # A genuine repeated local x-axis, using exactly the point-position map.
    cax.plot([j + 0.10, j + 0.10 + 0.78 * 0.8 / 0.85], [8.65, 8.65],
             color=MID, lw=0.5)
    for value in (0.0, 0.4, 0.8):
        xp = j + 0.10 + 0.78 * value / 0.85
        cax.plot([xp, xp], [-0.05, 8.4], color=GRID, lw=0.35, zorder=-2)
        cax.plot([xp, xp], [8.65, 8.78], color=MID, lw=0.5, clip_on=False)
        cax.text(xp, 8.97, f"{value:g}", ha="center", va="top", fontsize=7.0,
                 color=INK, clip_on=False).set_gid(f"hest-tick-{j}-{value:g}")

spec_order = ["NCBI783", "NCBI785", "TENX95", "TENX99", "TENX116", "TENX126", "TENX140", "TENX118", "TENX141"]
task_map = dict(HEST.drop_duplicates("specimen")[["specimen", "task"]].itertuples(index=False, name=None))
for i, specimen in enumerate(spec_order):
    task = task_map[specimen]
    if i % 2 == 0:
        cax.add_patch(Rectangle((-1.38, i - 0.45), 7.34, 0.90, facecolor="#F3F3F3", edgecolor="none", zorder=-3))
    cax.text(-0.16, i, specimen, ha="right", va="center", fontsize=7.0, color=INK)
    cax.plot([-0.11, -0.05], [i, i], color=TISSUE[task], lw=1.4, solid_capstyle="butt")
    rows = HEST[HEST["specimen"] == specimen].set_index("model")
    for j, metric in enumerate(metric_cols):
        v0 = float(rows.loc["ResNet50", metric])
        v1 = float(rows.loc["Phikon-v2", metric])
        x0 = j + 0.10 + 0.78 * v0 / 0.85
        x1 = j + 0.10 + 0.78 * v1 / 0.85
        cax.plot([x0, x1], [i, i], color="#aaa9a5", lw=0.65, zorder=1)
        cax.scatter(x0, i, s=11, facecolors="white", edgecolors=OPEN, linewidths=0.55, zorder=2).set_gid(f"hest-point-{i}-{j}-0")
        cax.scatter(x1, i, s=12, facecolors=PETROL, edgecolors=PETROL, linewidths=0.45, zorder=3).set_gid(f"hest-point-{i}-{j}-1")

groups = [("IDC", 0, 3), ("PAAD", 4, 6), ("LUNG", 7, 8)]
for task, lo, hi in groups:
    y0, y1 = lo - 0.37, hi + 0.37
    cax.plot([-1.20, -1.20], [y0, y1], color=TISSUE[task], lw=1.0)
    cax.plot([-1.20, -1.13], [y0, y0], color=TISSUE[task], lw=1.0)
    cax.plot([-1.20, -1.13], [y1, y1], color=TISSUE[task], lw=1.0)
    cax.text(-1.27, (lo + hi) / 2, task, ha="center", va="center", rotation=90,
             fontsize=7.0, color=TISSUE[task])

legend = [
    Line2D([0], [0], marker="o", markersize=4.0, markerfacecolor="white", markeredgecolor=OPEN,
           lw=0, label="ResNet50"),
    Line2D([0], [0], marker="o", markersize=4.0, markerfacecolor=PETROL, markeredgecolor=PETROL,
           lw=0, label="Phikon-v2"),
]
cax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1.0, 1.17), frameon=False,
           ncol=2, handletextpad=0.35, columnspacing=1.0, borderaxespad=0, fontsize=7.0)

texts = list(fig.texts)
for axis in fig.axes:
    texts += list(axis.texts) + [axis.title]
    if axis.axison:
        texts += list(axis.get_xticklabels()) + list(axis.get_yticklabels()) + [axis.xaxis.label, axis.yaxis.label]
    if axis.get_legend():
        texts += list(axis.get_legend().get_texts())
texts = [t for t in texts if t.get_visible() and t.get_text()]
assert min(t.get_fontsize() for t in texts) >= 7.0
for text in texts:
    text.set_zorder(20)
fig.set_dpi(300)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
failures = []
for t in texts:
    bb = t.get_window_extent(renderer)
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height:
        failures.append("clipped: " + t.get_text())
for i, t in enumerate(texts):
    for u in texts[i + 1:]:
        if t.get_window_extent(renderer).overlaps(u.get_window_extent(renderer)):
            failures.append(f"text overlap: {t.get_text()} / {u.get_text()}")
points = [p for p in cax.collections if (p.get_gid() or '').startswith('hest-point-')]
assert len(points) == 108
for point in points:
    i, j, arm = map(int, point.get_gid().split('-')[-3:])
    row = HEST[(HEST.specimen == spec_order[i]) & (HEST.model == ['ResNet50', 'Phikon-v2'][arm])].iloc[0]
    assert np.allclose(point.get_offsets()[0], [j + .10 + .78 * row[metric_cols[j]] / .85, i])
assert len([t for t in texts if (t.get_gid() or '').startswith('hest-tick-')]) == 18
fig.savefig(OUT / "figure_iter0.pdf", dpi=300)
fig.savefig(OUT / "img_iter0.png", dpi=300)
assert not failures, '\n'.join(failures)
(OUT / 'floor_selfcheck_iter0.txt').write_text(
    'overall: PASS\nsource_counts: CRC=768 P5_sections=128 P5_patient=64 HEST=18\n'
    'HEST_render: 108 points, all exact linear unclipped positions; 18 numerical ticks\n'
    'text_bboxes: disjoint and inside canvas\ntext_obscured_by_marks: PASS all text zorder=20\n'
    'minimum_visible_type: PASS 7 pt\nmetric_labels: PASS q is noise-corrected correlation; Cell composition and Pearson r unchanged\n'
    'decorative_Hest_row_fill: neutral #F3F3F3 only; scientific colors unchanged\n'
    'statistical_changes: none\npdf_fonttype: 42\n', encoding='utf-8')
plt.close(fig)
