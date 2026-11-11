02:18 ttft A/B start: package ~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-202611090700; new binary <fork> (07c045df); configs: cur-bal cur-thr new-off new-on; rounds 3; tests: probe4 probe8 agent4 mixed bench
02:18 power: 450.00 W 450.00 W 450.00 W 
cur-bal-r1 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
02:21 cur-bal-r1-probe4: probe c4 TTFT p50/p90/max 1.9784889221191406/2.5364670753479004/2.608963966369629 s (idle 0.29023051261901855); decoder tok/s free/during 698.4384983916085/597.4841175903715; max gap 0.2458477020263672 s
02:24 cur-bal-r1-probe8: probe c8 TTFT p50/p90/max 3.996016502380371/4.169564485549927/5.166186332702637 s (idle 0.29706382751464844); decoder tok/s free/during 1198.3380923158902/1108.4614797602321; max gap 0.2528676986694336 s
02:29 cur-bal-r1-agent4: agent c4: 424 turns in 300 s, TTFT p50/p90/max 1.5759754180908203/1.9708847999572754/3.022886037826538 s, 302.5 output tok/s
02:31 cur-bal-r1 mixed 3 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |cur-bal-r1-mixed3.json|)
02:33 cur-bal-r1 mixed 7 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |cur-bal-r1-mixed7.json|)
02:37 cur-bal-r1 bench: c1=302 c4=713 c8=-3
cur-thr-r1 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
02:41 cur-thr-r1-probe4: probe c4 TTFT p50/p90/max 1.5020155906677246/1.9314758777618408/2.0272316932678223 s (idle 0.30667543411254883); decoder tok/s free/during 656.3970588573263/537.398138521173; max gap 0.24811577796936035 s
02:43 cur-thr-r1-probe8: probe c8 TTFT p50/p90/max 2.7238967418670654/3.8824684619903564/4.001063108444214 s (idle 0.30173707008361816); decoder tok/s free/during 1142.2794348047269/1036.3016207480905; max gap 0.2611262798309326 s
02:49 cur-thr-r1-agent4: agent c4: 485 turns in 300 s, TTFT p50/p90/max 1.5749156475067139/2.031231164932251/2.8126277923583984 s, 242.5 output tok/s
02:51 cur-thr-r1 mixed 3 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |cur-thr-r1-mixed3.json|)
02:53 cur-thr-r1 mixed 7 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |cur-thr-r1-mixed7.json|)
02:56 cur-thr-r1 bench: c1=302 c4=721 c8=985
new-off-r1 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
03:00 new-off-r1-probe4: probe c4 TTFT p50/p90/max 2.6009726524353027/2.711026906967163/2.94633150100708 s (idle 0.3113386631011963); decoder tok/s free/during 724.0969737491814/629.2207895908529; max gap 0.25008058547973633 s
03:03 new-off-r1-probe8: probe c8 TTFT p50/p90/max 4.83912205696106/5.001981258392334/5.025302410125732 s (idle 0.3180060386657715); decoder tok/s free/during 1105.123136541698/1007.9101391453928; max gap 0.25576186180114746 s
03:08 new-off-r1-agent4: agent c4: 380 turns in 300 s, TTFT p50/p90/max 1.6203556060791016/2.033252239227295/3.6060361862182617 s, 339.1 output tok/s
03:10 new-off-r1 mixed 3 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |new-off-r1-mixed3.json|)
03:12 new-off-r1 mixed 7 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |new-off-r1-mixed7.json|)
03:16 new-off-r1 bench: c1=298 c4=703 c8=969
new-on-r1 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
03:20 new-on-r1-probe4: probe c4 TTFT p50/p90/max 0.36400938034057617/0.37472963333129883/0.38117527961730957 s (idle 0.29499363899230957); decoder tok/s free/during 724.1629116759996/190.31470597759923; max gap 0.3126816749572754 s
03:22 new-on-r1-probe8: probe c8 TTFT p50/p90/max 0.3741466999053955/0.3807833194732666/0.3873918056488037 s (idle 0.3009781837463379); decoder tok/s free/during 1109.0237158145/308.77903238067574; max gap 0.3026773929595947 s
03:28 new-on-r1-agent4: agent c4: 507 turns in 300 s, TTFT p50/p90/max 0.5332410335540771/0.8651919364929199/1.4510722160339355 s, 339.8 output tok/s
03:29 new-on-r1 mixed 3 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |new-on-r1-mixed3.json|)
03:31 new-on-r1 mixed 7 streams: exit 0 (length on an idle server; TTFT load = a unique prompt sent after the background |warmup; slowdown = TTFT load / TTFT alone. bg before / prefill / after = |new-on-r1-mixed7.json|)
03:35 new-on-r1 bench: c1=300 c4=717 c8=976
cur-bal-r2 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
03:39 cur-bal-r2-probe4: probe c4 TTFT p50/p90/max 2.0397841930389404/2.0763158798217773/2.085477590560913 s (idle 0.2948181629180908); decoder tok/s free/during 715.3885364481677/608.7106122325141; max gap 0.24913787841796875 s
03:41 cur-bal-r2-probe8: probe c8 TTFT p50/p90/max 3.9552969932556152/5.1141040325164795/5.215575456619263 s (idle 0.3230102062225342); decoder tok/s free/during 1186.5869766111189/1093.6129044335726; max gap 0.2566184997558594 s
03:47 cur-bal-r2-agent4: agent c4: 456 turns in 300 s, TTFT p50/p90/max 1.6538424491882324/1.9745852947235107/3.0032567977905273 s, 260.0 output tok/s
03:51 cur-bal-r2 bench: c1=292 c4=687 c8=942
cur-thr-r2 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
03:55 cur-thr-r2-probe4: probe c4 TTFT p50/p90/max 1.55072021484375/2.136545419692993/2.146599531173706 s (idle 0.29865431785583496); decoder tok/s free/during 727.5168889096302/593.8523639538256; max gap 0.45407819747924805 s
03:57 cur-thr-r2-probe8: probe c8 TTFT p50/p90/max 2.828378200531006/3.8000576496124268/3.90193772315979 s (idle 0.302288293838501); decoder tok/s free/during 1146.145573140902/1035.148424182782; max gap 0.2586216926574707 s
04:03 cur-thr-r2-agent4: agent c4: 474 turns in 300 s, TTFT p50/p90/max 1.58103609085083/1.9440038204193115/2.974400758743286 s, 246.0 output tok/s
04:06 cur-thr-r2 bench: c1=291 c4=695 c8=985
new-off-r2 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
04:10 new-off-r2-probe4: probe c4 TTFT p50/p90/max 2.083381175994873/2.6357507705688477/2.678696870803833 s (idle 0.31325554847717285); decoder tok/s free/during 724.616487775545/626.0031359240663; max gap 0.24943828582763672 s
04:13 new-off-r2-probe8: probe c8 TTFT p50/p90/max 3.822524309158325/5.076319456100464/5.100054979324341 s (idle 0.30092954635620117); decoder tok/s free/during 1093.3290252208835/1026.494736445628; max gap 0.2725245952606201 s
04:18 new-off-r2-agent4: agent c4: 423 turns in 300 s, TTFT p50/p90/max 1.585437536239624/1.9454820156097412/3.145402431488037 s, 282.1 output tok/s
04:22 new-off-r2 bench: c1=303 c4=702 c8=988
new-on-r2 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
04:26 new-on-r2-probe4: probe c4 TTFT p50/p90/max 0.3629486560821533/0.3725881576538086/0.38054823875427246 s (idle 0.2975351810455322); decoder tok/s free/during 732.9134269564191/191.6763084508228; max gap 0.29937076568603516 s
04:28 new-on-r2-probe8: probe c8 TTFT p50/p90/max 0.37819814682006836/0.3847057819366455/0.3903789520263672 s (idle 0.30281567573547363); decoder tok/s free/during 1195.1087595753697/333.9483467744032; max gap 0.2699775695800781 s
04:34 new-on-r2-agent4: agent c4: 538 turns in 300 s, TTFT p50/p90/max 0.5485837459564209/0.9992311000823975/1.5003268718719482 s, 270.1 output tok/s
04:38 new-on-r2 bench: c1=295 c4=708 c8=965
cur-bal-r3 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
04:41 cur-bal-r3-probe4: probe c4 TTFT p50/p90/max 2.063096761703491/2.6710944175720215/2.972036361694336 s (idle 0.3161447048187256); decoder tok/s free/during 726.0601612742661/626.5614395701848; max gap 0.25005555152893066 s
04:44 cur-bal-r3-probe8: probe c8 TTFT p50/p90/max 4.073110580444336/5.238160610198975/5.267293453216553 s (idle 0.31487321853637695); decoder tok/s free/during 1177.4283174324878/1095.4512406870292; max gap 0.2557871341705322 s
04:49 cur-bal-r3-agent4: agent c4: 448 turns in 300 s, TTFT p50/p90/max 1.614595651626587/1.9623758792877197/3.4542300701141357 s, 271.4 output tok/s
04:53 cur-bal-r3 bench: c1=302 c4=716 c8=968
cur-thr-r3 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
04:57 cur-thr-r3-probe4: probe c4 TTFT p50/p90/max 1.5695078372955322/1.6327321529388428/2.0837461948394775 s (idle 0.3049943447113037); decoder tok/s free/during 722.1503259403254/578.692238945852; max gap 0.4572453498840332 s
04:59 cur-thr-r3-probe8: probe c8 TTFT p50/p90/max 2.906682014465332/3.860328197479248/3.9015462398529053 s (idle 0.31390857696533203); decoder tok/s free/during 1171.755873935424/1062.0755288792816; max gap 0.23482656478881836 s
05:05 cur-thr-r3-agent4: agent c4: 453 turns in 300 s, TTFT p50/p90/max 1.5225317478179932/1.894916296005249/6.226598739624023 s, 268.3 output tok/s
05:09 cur-thr-r3 bench: c1=286 c4=696 c8=985
new-off-r3 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
05:13 new-off-r3-probe4: probe c4 TTFT p50/p90/max 2.0659406185150146/2.4157016277313232/2.7841532230377197 s (idle 0.2985811233520508); decoder tok/s free/during 723.6092594509844/614.3270613965966; max gap 0.4563462734222412 s
05:15 new-off-r3-probe8: probe c8 TTFT p50/p90/max 3.991253137588501/5.203020095825195/5.679847717285156 s (idle 0.3020172119140625); decoder tok/s free/during 1189.8474259521367/1100.8954093421066; max gap 0.2576029300689697 s
05:21 new-off-r3-agent4: agent c4: 449 turns in 300 s, TTFT p50/p90/max 1.6210541725158691/2.000471830368042/3.0444087982177734 s, 247.7 output tok/s
05:25 new-off-r3 bench: c1=292 c4=737 c8=958
new-on-r3 KV tokens · 285 shared snapshot slots · limited by memory  0.2s
05:28 new-on-r3-probe4: probe c4 TTFT p50/p90/max 0.3654210567474365/0.37340831756591797/0.4729776382446289 s (idle 0.2925374507904053); decoder tok/s free/during 729.7011504528368/192.64446629450032; max gap 0.4004514217376709 s
05:31 new-on-r3-probe8: probe c8 TTFT p50/p90/max 0.3783559799194336/0.38418149948120117/0.38907909393310547 s (idle 0.2998020648956299); decoder tok/s free/during 1183.7801586564278/327.24574957742436; max gap 0.3096425533294678 s
05:36 new-on-r3-agent4: agent c4: 519 turns in 300 s, TTFT p50/p90/max 0.5477173328399658/0.9267621040344238/1.5328741073608398 s, 294.3 output tok/s
05:40 new-on-r3 bench: c1=307 c4=707 c8=964
| metric (median over rounds) | cur-bal | cur-thr | new-off | new-on |
|---|---|---|---|---|
| agent4 p90 | 1.97 (n=3) | 1.94 (n=3) | 2 (n=3) | 0.927 (n=3) |
| agent4 tok/s | 271 (n=3) | 246 (n=3) | 282 (n=3) | 294 (n=3) |
| agent4 turns | 448 (n=3) | 474 (n=3) | 423 (n=3) | 519 (n=3) |
| bench c1 | 302 (n=3) | 291 (n=3) | 298 (n=3) | 300 (n=3) |
| bench c4 | 713 (n=3) | 696 (n=3) | 703 (n=3) | 708 (n=3) |
| bench c8 | 942 (n=3) | 985 (n=3) | 969 (n=3) | 965 (n=3) |
| probe4 dec free | 715 (n=3) | 722 (n=3) | 724 (n=3) | 730 (n=3) |
| probe4 gap | 0.249 (n=3) | 0.454 (n=3) | 0.25 (n=3) | 0.313 (n=3) |
| probe4 p50 | 2.04 (n=3) | 1.55 (n=3) | 2.08 (n=3) | 0.364 (n=3) |
| probe4 p90 | 2.54 (n=3) | 1.93 (n=3) | 2.64 (n=3) | 0.373 (n=3) |
| probe8 dec free | 1.19e+03 (n=3) | 1.15e+03 (n=3) | 1.11e+03 (n=3) | 1.18e+03 (n=3) |
| probe8 gap | 0.256 (n=3) | 0.259 (n=3) | 0.258 (n=3) | 0.303 (n=3) |
| probe8 p50 | 4 (n=3) | 2.83 (n=3) | 3.99 (n=3) | 0.378 (n=3) |
| probe8 p90 | 5.11 (n=3) | 3.86 (n=3) | 5.08 (n=3) | 0.384 (n=3) |
05:41 default restored; ttft A/B done
