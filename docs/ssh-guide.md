## 1 一句话概括

SSH 密钥认证 = **锁 + 钥匙**。私钥是钥匙（留本机），公钥是锁（放 GitHub）。GitHub 看到你用匹配的钥匙开锁，就放行。

---

## 2 密钥对原理

```
┌─────────────────────────────────────────────────────────────────┐
│  ssh-keygen 生成一对密钥，它们数学上匹配但无法互相推导           │
│                                                                 │
│  私钥  ~/.ssh/id_ed25519   你本机的文件，绝不外传               │
│  公钥  ~/.ssh/id_ed25519.pub  可以随便分发，贴到 GitHub 上       │
└─────────────────────────────────────────────────────────────────┘
```

连接时：

```
你本机                              GitHub
──────                              ──────
"我是 wens" 
    ←── 随机数挑战 ────────────────
用私钥签名该随机数
    ──── 签名结果 ───────────────→  用你之前贴的公钥验证
                                    签名对了 → 放行
                                    签名错了 → Permission denied
```

全程私钥没离开过你的电脑，GitHub 也没见到你的私钥。这就是非对称加密的核心。

---

## 3 每个机子需要一个独立的私钥

**正确做法**：一台机器一个密钥对。

```
你的 MacBook               你的台式机                 你的服务器
~/.ssh/id_ed25519          ~/.ssh/id_ed25519         ~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub      ~/.ssh/id_ed25519.pub     ~/.ssh/id_ed25519.pub
      │                          │                          │
      └────────── 都贴到 GitHub Settings ──────────────────┘
                 https://github.com/settings/keys
```

**为什么**：如果私钥泄露（比如笔记本丢了），你只需要在 GitHub 上删掉那一台机器的公钥，其他机器不受影响。如果所有机器共用一个私钥，泄露后全部得换。

GitHub 支持添加**多个** SSH Key，每个 Key 可以标注是哪台机器（如 "MacBook"、"台式机"、"服务器"）。

---

## 4 实际操作演示

### 4.1 生成密钥（每台新机器做一次）

```bash
# 你的 MacBook 上刚才执行的
ssh-keygen -t ed25519 -C "wens@buaa" -f ~/.ssh/id_ed25519 -N ""

# 参数说明：
#   -t ed25519   使用 Ed25519 算法（比 RSA 更快更安全）
#   -C "备注"     注释，方便在 GitHub 上辨认是哪台机器
#   -f 路径       私钥存放位置
#   -N ""        密码为空（生产环境建议设密码）
```

### 4.2 查看公钥

```bash
cat ~/.ssh/id_ed25519.pub
# 输出：ssh-ed25519 AAAAC3Nz... wens@buaa
```

### 4.3 贴到 GitHub

打开 `github.com/settings/keys` → New SSH Key：
- Title：`MacBook Air`（自己起名，方便以后辨认）
- Key：粘贴上面 cat 出来的整行内容

### 4.4 验证连接

```bash
ssh -T git@github.com
# 成功：Hi Ding141! You've successfully authenticated.
# 失败：Permission denied (publickey).
```

### 4.5 Git 使用 SSH 远端

```bash
# HTTPS 方式（已被校园网拦截）
# https://github.com/Ding141/repo.git

# SSH 方式
git remote set-url origin git@github.com:Ding141/repo.git
git push   # 自动走 SSH，无需密码
```

---

## 5 常见问题

**Q: 私钥丢了怎么办？**
在 GitHub Settings 里删掉对应的公钥，生成新的。

**Q: 换了新电脑？**
新电脑上生成新密钥 → 公钥贴到 GitHub → 旧电脑的公钥删掉（安全起见）。

**Q: 可以给多台机器用同一个私钥吗？**
技术上可以（复制 `~/.ssh/id_ed25519` 到其他机器），但**不推荐**。一台泄露全部沦陷。

**Q: 重装系统后？**
`~/.ssh/` 会被清掉，需要重新生成并重新贴公钥。

---

## 6 和你这次的关联

刚才发生的事情：

```
1. 本机 HTTPS push GitHub → 校园网拦截 443 端口 → 超时

2. 检测 SSH 22 端口 → 能通！

3. ssh-keygen 生成密钥对 → 私钥留本机

4. 你把公钥贴到 GitHub → GitHub 信任这把"锁"

5. git push → Git 走 SSH 22 端口 → GitHub 验签通过 → 推送成功
```

以后本机 push 直接 `git push` 即可，不用再输密码，也不用担心校园网拦截 HTTPS。
