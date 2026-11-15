import io, re, shutil, sys
import sys
src = sys.argv[1] if len(sys.argv) > 1 else "dlrm_v2_train.py"
dst = sys.argv[2] if len(sys.argv) > 2 else "dlrm_v2_train_opt.py"
s = open(src, encoding="utf-8").read()
orig = s

# 1) 新增 --sync-every
s = s.replace('    p.add_argument("--csv", type=str, default="", help="把每步耗时写到 csv")',
              '    p.add_argument("--csv", type=str, default="", help="把每步耗时写到 csv")\n'
              '    p.add_argument("--sync-every", type=int, default=50,\n'
              '                   help="P2：每 N 步同步一次（1=原行为，逐步同步）")', 1)

# 2) 计时锚点初始化
s = s.replace("    step_global = 0\n", "    step_global = 0\n    _t_last = time.time()\n    _g_last = 0\n", 1)

# 3) 训练步：去掉逐步同步与标量回读
old = '''                torch.npu.synchronize()
                dt = time.time() - t0
                step_times.append(dt)
                losses.append(loss.item())

                with torch.no_grad():
                    acc = float(((torch.sigmoid(logits) > 0.5).float() == label)
                                .float().mean().item())
                accs.append(acc)

                if is_main and (step_global < 3 or (step_global + 1) % 50 == 0):
                    print("ep%d step %5d/%d  loss=%.6f acc=%.4f  %.2f ms"
                          % (ep, bi + 1, nb, loss.item(), acc, dt * 1000), flush=True)
'''
new = '''                _do = (args.sync_every <= 1) or (step_global < 3) \\
                      or (step_global % args.sync_every == 0)
                if _do:
                    torch.npu.synchronize()
                    _now = time.time()
                    _k = step_global - _g_last
                    if _k > 0:
                        step_times.append((_now - _t_last) / _k)
                    _t_last, _g_last = _now, step_global
                    _lv = float(loss.detach().item())
                    losses.append(_lv)
                    if is_main and (step_global < 3 or (step_global + 1) % 50 == 0
                                    or step_global + 1 == args.steps):
                        with torch.no_grad():
                            acc = float(((torch.sigmoid(logits.detach()) > 0.5).float() == label)
                                        .float().mean().item())
                        accs.append(acc)
                        print("ep%d step %5d/%d  loss=%.6f acc=%.4f  %.2f ms"
                              % (ep, bi + 1, nb, _lv, acc, step_times[-1] * 1000), flush=True)
'''
assert old in s, "训练循环片段未匹配"
s = s.replace(old, new, 1)
assert s != orig
open(dst, "w", encoding="utf-8").write(s)
print("written", dst, "len", len(s), "sync-every 出现", s.count("sync_every"))
