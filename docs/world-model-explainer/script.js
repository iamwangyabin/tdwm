const pipelineContent = [
  {
    title: "离线数据记录发生过什么",
    description: "每条 episode 保存连续图像、执行动作以及用于恢复环境的状态。它只覆盖专家或采集策略走过的有限轨迹，并不穷举所有可能状态。",
    note: "数据是经验样本，不是游戏世界本身。",
    visual: "dataset"
  },
  {
    title: "模型从相邻片段学习动作后果",
    description: "LeWM 把图像编码到 latent，用历史帧和动作预测稍后的 latent；RC-aux 或长期 TD 路线会额外要求表示携带多步可达信息。",
    note: "训练时看见的是数据轨迹，模型要学的是可泛化的动力规律。",
    visual: "training"
  },
  {
    title: "数据集只负责出一道可重复的题",
    description: "评测程序从某条 episode 取一个起始状态，再把之后的状态当作目标。真实环境被恢复到起点，原示范动作并不会交给待测模型。",
    note: "同一批起点和目标让不同方法面对完全相同的难度。",
    visual: "exam"
  },
  {
    title: "规划器让模型想象许多种未来",
    description: "CEM 采样数百组候选动作序列，世界模型预测各自未来，并给出目标代价。这里的 rollout 只是模型内部计算，不会直接改变真实环境。",
    note: "LeWM 当前主协议是 300 candidates、30 elites、horizon 5。",
    visual: "imagine"
  },
  {
    title: "只执行一小段，然后重新观察",
    description: "选中的动作进入 Pymunk、MuJoCo 或二维碰撞模拟器。环境计算真正的碰撞、摩擦和关节变化，再渲染新画面，规划器据此重新规划。",
    note: "模型预测错了，真实环境不会配合它，错误会直接暴露。",
    visual: "execute"
  },
  {
    title: "最终由环境统计成功或失败",
    description: "每局达到环境定义的目标阈值才算成功，同时保存逐 episode 成败、种子、耗时和必要的视频。平均成功率只是汇总，不应替代原始结果。",
    note: "不能让世界模型自己宣布成功，裁判必须是环境。",
    visual: "score"
  }
];

const environmentContent = {
  tworoom: {
    index: "01 / 04",
    image: "assets/tworoom.gif",
    alt: "TwoRoom 环境运行画面",
    engine: "Torch 二维碰撞",
    id: "swm/TwoRoom-v1",
    title: "穿过门，而不是撞向墙",
    description: "圆形智能体从一个房间出发，到另一个房间的目标点。直线距离看起来很近，但真正可行的路线必须穿过门洞。",
    action: "二维速度方向",
    success: "距离目标小于 16 像素",
    test: "拓扑、绕障与长期可达性",
    risk: "latent 距离近，不代表穿墙可达"
  },
  reacher: {
    index: "02 / 04",
    image: "assets/reacher.gif",
    alt: "Reacher 机械臂环境运行画面",
    engine: "MuJoCo",
    id: "swm/ReacherDMControl-v0",
    title: "控制关节，而不是拖动指尖",
    description: "二维两连杆机械臂接收两个关节力矩。在本项目的 qpos-match 任务中，它必须从当前关节状态到达目标关节配置。",
    action: "两个连续关节力矩",
    success: "达到目标关节配置阈值",
    test: "连续动力学、精确控制与状态恢复",
    risk: "末端接近目标不等于关节配置正确"
  },
  pusht: {
    index: "03 / 04",
    image: "assets/pusht.gif",
    alt: "PushT 环境运行画面",
    engine: "Pymunk / 10 Hz",
    id: "swm/PushT-v1",
    title: "接触发生之后，误差会放大",
    description: "圆形末端执行器推动 T 形物体，同时要匹配目标位置和旋转角。一次推偏可能改变后续所有接触关系。",
    action: "二维相对速度",
    success: "位置误差 < 20 px 且角度误差 < 20°",
    test: "接触动力学、旋转与多步动作顺序",
    risk: "长期 temporal cost 可能牺牲局部接触几何"
  },
  cube: {
    index: "04 / 04",
    image: "assets/cube.gif",
    alt: "OGBench Cube 机器人环境运行画面",
    engine: "OGBench + MuJoCo",
    id: "swm/OGBCube-v0",
    title: "七维控制中的抓取与放置",
    description: "UR5e 机械臂与 Robotiq 夹爪移动一个方块到目标位置。视觉、机械臂本体状态、抓取时机和三维接触都影响结果。",
    action: "6 个关节速度 + 夹爪",
    success: "方块距目标小于 4 cm",
    test: "三维操作、抓取和高维动作搜索",
    risk: "数据和模型开销最大，规划对误差很敏感"
  }
};

const methodContent = {
  lewm: {
    family: "局部、动作条件世界模型",
    title: "LeWM：先学会预测下一段 latent",
    summary: "从像素编码当前状态，根据任意候选动作预测未来 latent，再用 latent 距离作为目标代价，让 CEM 搜索动作序列。",
    flow: ["图像", "Encoder", "Latent", "Action", "Predictor", "未来 latent"],
    pros: ["reward-free，训练数据无需任务奖励", "能比较任意候选 primitive action", "结构简单，适合做干净、可审计的基线"],
    cons: ["一步预测准，不保证多步规划准", "latent L2 可能不等于真实可达距离", "开放环滚动会累积误差"],
    verdict: "所有增强方法都必须打败的公平底座"
  },
  rcaux: {
    family: "多步 rollout + reachability 辅助",
    title: "RC-aux：让表示知道“能不能到”",
    summary: "保留 LeWM 的编码器、预测器和防坍塌约束，同时加入真实轨迹上的递归 rollout、多 horizon 预测、可达性或距离监督，使 latent 更贴近规划需要。",
    flow: ["LeWM", "多步 rollout", "Reachability head", "可达代价", "MPC"],
    pros: ["直接针对“predictive but not plannable”", "不必依赖外部任务奖励", "保留 LeWM 的任意动作 MPC 接口"],
    cons: ["示范步数只是最短路径的上界", "跨轨迹负样本可能其实可达", "PushT 等接触任务可能不受益甚至退化"],
    verdict: "最直接的长期可达增强基线"
  },
  tdjepa: {
    family: "长期 TD / policy-conditioned 表示",
    title: "TD-JEPA：用 TD 传播长期未来信息",
    summary: "不只拟合一步未来，而是通过时序差分把远期、策略条件的未来占用或价值结构向当前表示传播，强调长期决策与 zero-shot 控制。",
    flow: ["状态", "策略条件", "TD target", "长期表征", "Zero-shot policy"],
    pros: ["长时信用分配更直接", "能表达 policy-conditioned future", "在长期目标控制上具有强竞争力"],
    cons: ["不一定保留任意动作反事实预测", "容易绑定训练策略或任务分布", "与 LeWM 的 MPC 接口并非天然等价"],
    verdict: "长期控制表征的强邻近方法，而非简单 LeWM 插件"
  }
};

const pipelineButtons = document.querySelectorAll(".pipeline-step");
const stageVisual = document.querySelector(".stage-visual");
pipelineButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const index = Number(button.dataset.step);
    const content = pipelineContent[index];
    pipelineButtons.forEach((item) => {
      item.classList.toggle("active", item === button);
      item.setAttribute("aria-selected", item === button ? "true" : "false");
    });
    document.getElementById("step-number").textContent = `${String(index + 1).padStart(2, "0")} / 06`;
    document.getElementById("step-title").textContent = content.title;
    document.getElementById("step-description").textContent = content.description;
    document.getElementById("step-note").innerHTML = `<strong>关键：</strong>${content.note}`;
    stageVisual.dataset.stageVisual = content.visual;
  });
});

document.querySelectorAll(".environment-tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    const content = environmentContent[button.dataset.env];
    document.querySelectorAll(".environment-tabs button").forEach((item) => {
      item.classList.toggle("active", item === button);
      item.setAttribute("aria-selected", item === button ? "true" : "false");
    });
    const image = document.getElementById("env-image");
    image.src = content.image;
    image.alt = content.alt;
    document.getElementById("env-engine").textContent = content.engine;
    document.getElementById("env-id").textContent = content.id;
    document.getElementById("env-index").textContent = content.index;
    document.getElementById("env-title").textContent = content.title;
    document.getElementById("env-description").textContent = content.description;
    document.getElementById("env-action").textContent = content.action;
    document.getElementById("env-success").textContent = content.success;
    document.getElementById("env-test").textContent = content.test;
    document.getElementById("env-risk").textContent = content.risk;
  });
});

document.querySelectorAll(".mode-switch button").forEach((button) => {
  button.addEventListener("click", () => {
    const isDataset = button.dataset.mode === "dataset";
    document.querySelectorAll(".mode-switch button").forEach((item) => item.classList.toggle("active", item === button));
    document.getElementById("demo-mode-label").textContent = isDataset ? "Dataset-driven" : "Random reset";
    document.getElementById("mode-title").textContent = isDataset ? "固定的是起点和目标，不是路线" : "随机模式更自由，但方差也更大";
    document.getElementById("mode-copy").textContent = isDataset
      ? "从一条已经完成的轨迹中取出起点 S 和稍后的目标 G，因此知道任务可达。模型可以走原路线、捷径或完全不同的路线；物理环境只看它最终是否达到 G。"
      : "环境按 seed 随机生成起点和目标，模型依旧自由控制。它适合测总体鲁棒性，但不同关卡难度波动更大，方法之间必须严格共享 seed 才可比较。";
    document.querySelector(".demo-footer").innerHTML = isDataset
      ? "<span>起点：数据轨迹第 10 步</span><span>目标：第 35 步状态</span><span>预算：50 步</span>"
      : "<span>起点：环境随机采样</span><span>目标：环境随机采样</span><span>seed：严格固定</span>";
    document.getElementById("demo-path").setAttribute("d", isDataset
      ? "M90,185 C190,220 240,205 310,132 C380,62 475,72 560,92"
      : "M90,185 C175,105 220,86 310,132 C405,188 485,155 560,92");
  });
});

function renderMethodFlow(items) {
  return items.map((item, index) => {
    const separator = index === 0 ? "" : "<i>→</i>";
    return `${separator}<span>${item}</span>`;
  }).join("");
}

document.querySelectorAll(".method-picker button").forEach((button) => {
  button.addEventListener("click", () => {
    const content = methodContent[button.dataset.method];
    document.querySelectorAll(".method-picker button").forEach((item) => {
      item.classList.toggle("active", item === button);
      item.setAttribute("aria-selected", item === button ? "true" : "false");
    });
    document.getElementById("method-family").textContent = content.family;
    document.getElementById("method-title").textContent = content.title;
    document.getElementById("method-summary").textContent = content.summary;
    document.getElementById("method-flow").innerHTML = renderMethodFlow(content.flow);
    document.getElementById("method-pros").innerHTML = content.pros.map((item) => `<li>${item}</li>`).join("");
    document.getElementById("method-cons").innerHTML = content.cons.map((item) => `<li>${item}</li>`).join("");
    document.getElementById("method-verdict").textContent = content.verdict;
  });
});

const menuButton = document.querySelector(".menu-button");
const nav = document.querySelector(".nav");
menuButton.addEventListener("click", () => {
  const isOpen = nav.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", isOpen ? "true" : "false");
});
nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
  nav.classList.remove("open");
  menuButton.setAttribute("aria-expanded", "false");
}));

const sections = [...document.querySelectorAll("main section[id]")];
const navLinks = [...document.querySelectorAll(".nav a")];
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    navLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`));
  });
}, { rootMargin: "-35% 0px -55% 0px" });
sections.forEach((section) => observer.observe(section));
