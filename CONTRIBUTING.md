# 协作指南 (Contributing Guide)

## 首次设置

### 1. 克隆仓库

```bash
git clone https://github.com/xz943-a11y/fixed-income-agents.git
cd fixed-income-agents
```

### 2. 配置 Git 用户信息

```bash
git config --global user.name "你的名字"
git config --global user.email "your.email@example.com"
```

### 3. 创建 Personal Access Token

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置：
   - Note: `fixed-income-agents`
   - Expiration: 选择过期时间（建议 90 天或更长）
   - 勾选 `repo` 权限
4. 点击 "Generate token"
5. **复制并保存 token**（只显示一次！）

### 4. 配置 Credential Helper（macOS）

```bash
git config --global credential.helper osxkeychain
```

### 5. 测试推送权限

```bash
# 修改任意文件或创建新文件
echo "# Test" >> test.txt
git add test.txt
git commit -m "Test commit"
git push origin main
```

首次推送时会提示输入：
- **Username**: 你的 GitHub 用户名
- **Password**: 粘贴你的 Personal Access Token（不是 GitHub 密码）

## 日常协作流程

### 1. 拉取最新代码

```bash
git pull origin main
```

### 2. 创建新功能分支（推荐）

```bash
git checkout -b feature/your-feature-name
# 或者
git checkout -b fix/bug-description
```

### 3. 提交更改

```bash
git add .
git commit -m "描述你的更改"
git push origin your-branch-name
```

### 4. 在主分支上工作（如果直接在主分支）

```bash
git add .
git commit -m "描述你的更改"
git push origin main
```

## 注意事项

- 推送前先 `git pull` 确保代码是最新的
- 提交信息要清晰描述更改内容
- 建议使用分支进行开发，完成后合并到 main
- 不要提交敏感信息（API keys、密码等）

## 需要帮助？

如果遇到权限问题，请联系仓库管理员添加你为协作者。

