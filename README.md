<<<<<<< HEAD
<h1 align="center">MiroTalk P2P</h1>

<h3 align="center">Open Source WebRTC P2P Video Conferencing You Can Self-Host in Minutes</h3>

<h4 align="center">Free, Secure, Fast Real-Time Communication - up to 8K, 60fps. Works in All Browsers and Platforms.</h4>

<br />

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/miroslavpejic85/mirotalk?style=social)](https://github.com/miroslavpejic85/mirotalk/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/miroslavpejic85/mirotalk?style=social)](https://github.com/miroslavpejic85/mirotalk/network/members)

<a href="https://choosealicense.com/licenses/agpl-3.0/">![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3_Open_Source-blue.svg)</a>
<a href="https://hub.docker.com/r/mirotalk/p2p">![Docker Pulls](https://img.shields.io/docker/pulls/mirotalk/p2p)</a>
<a href="https://github.com/miroslavpejic85/mirotalk/commits/master">![Last Commit](https://img.shields.io/github/last-commit/miroslavpejic85/mirotalk)</a>
<a href="https://discord.gg/rgGYfeYW3N">![Discord](https://img.shields.io/badge/Discord-Community-5865F2?logo=discord&logoColor=white)</a>
<a href="https://www.linkedin.com/in/miroslav-pejic-976a07101/">![Author](https://img.shields.io/badge/Author-Miroslav_Pejic-brightgreen.svg)</a>

</div>

<br />

<p align="center"><strong>MiroTalk P2P</strong> is a <strong>self-hosted, open-source video conferencing</strong> platform using direct <strong>peer-to-peer WebRTC connections</strong> for fast, secure, real-time communication. Deploy on your own server in minutes. Enjoy unlimited rooms, no time limits, end-to-end privacy, and a rich feature set - all under your control.</p>

<p align="center">
    <a href="https://p2p.mirotalk.com">Try Live Demo</a> · <a href="https://p2p.mirotalk.com/privacy">Privacy</a> · <a href="https://docs.mirotalk.com/mirotalk-p2p/self-hosting/">Documentation</a> · <a href="https://discord.gg/rgGYfeYW3N">Discord</a> · <a href="https://github.com/sponsors/miroslavpejic85">Sponsor</a>
</p>

<br />

<p align="center">
    <a href="https://p2p.mirotalk.com/">
        <img src="public/images/mirotalk-github.gif" alt="MiroTalk P2P - Open Source Video Conferencing">
    </a>
</p>

<p align="center">Proudly sponsored by</p>

<h1 align=center><a href="https://www.recall.ai/?utm_source=github&utm_medium=sponsorship&utm_campaign=miroslavpejic85-mirotalk">Recall.ai</a> - API for meeting recording</h1>
<p align="center">An API for recording Zoom, Google Meet, Microsoft Teams, and in-person meetings.</p>

<hr />

<br />

<details>
<summary>✨ Why MiroTalk P2P?</summary>

<br/>

|                    | MiroTalk P2P                                                                                                                                        | Other Solutions             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| 💰 **Cost**        | Free & Open Source (AGPLv3). [One-time fee licenses](https://codecanyon.net/item/mirotalk-p2p-webrtc-realtime-video-conferences/38376661) available | Paid plans                  |
| 🏠 **Self-hosted** | ✅ Full control over your data                                                                                                                      | ❌ Cloud only               |
| 🔒 **Privacy**     | Your server, your rules                                                                                                                             | Third-party data processing |
| ⏱️ **Time limits** | Unlimited                                                                                                                                           | 40-60 min on free tiers     |
| 🏢 **Rooms**       | Unlimited concurrent rooms                                                                                                                          | Limited                     |
| 🎥 **Resolution**  | Up to 8K @ 60fps                                                                                                                                    | Up to 1080p                 |
| 🌍 **Languages**   | 133 languages                                                                                                                                       | ~30-80                      |
| 🔌 **API**         | Full REST API included                                                                                                                              | Paid add-on                 |
| 🤖 **AI Features** | ChatGPT (OpenAI) integration                                                                                                                        | Paid AI add-ons             |
| 🧩 **Rebrand**     | Full source code, white-label ready                                                                                                                 | Limited branding options    |
| 📦 **Deploy**      | Docker, Node.js, one-click install                                                                                                                  | N/A (SaaS only)             |

</details>

<details>
<summary>🚀 Features</summary>

<br/>

- 🎥 Video up to **8K @ 60fps** · Screen sharing · Recording · Picture-in-Picture
- 💬 Chat with Markdown & emoji · Collaborative whiteboard · File sharing
- 🤖 ChatGPT (OpenAI) integration · Speech recognition
- 🔒 OIDC auth · [Host protection](https://docs.mirotalk.com/mirotalk-p2p/host-protection/) · JWT credentials · Room passwords · Peer-to-peer encryption
- 🔌 REST API · Slack & Mattermost · Embeddable [iframe](https://docs.mirotalk.com/mirotalk-p2p/integration/#iframe) & [widget](https://docs.mirotalk.com/mirotalk-p2p/integration/#widgets-integration) · 133 languages

**[See all features →](https://docs.mirotalk.com/overview/)**

</details>

<details open>
<summary>⚡ Quick start</summary>

<br/>

**Start in 6 commands:**

```bash
git clone https://github.com/miroslavpejic85/mirotalk.git
cd mirotalk
cp .env.template .env
cp app/src/config.template.js app/src/config.js
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) - done!

</details>

<details>
<summary>🐳 Docker</summary>

<br/>

![docker](public/images/docker.png)

**Prerequisites:** Install [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) - Image available on [Docker Hub](https://hub.docker.com/r/mirotalk/p2p)

```bash
git clone https://github.com/miroslavpejic85/mirotalk.git
cd mirotalk
cp .env.template .env
cp app/src/config.template.js app/src/config.js
cp docker-compose.template.yml docker-compose.yml
docker-compose pull    # optional: pull official image
docker-compose up      # add -d to run in background
```

Open [http://localhost:3000](http://localhost:3000) - done!

> **Note:**
> Edit `app/src/config.js`, `.env`, and `docker-compose.yml` to customize your setup.

</details>

<details>
<summary>📚 Documentation</summary>

<br/>

For detailed guides and references, visit the **[official documentation](https://docs.mirotalk.com)**:

- [Our Story](https://docs.mirotalk.com/story/)
- [About](https://docs.mirotalk.com/mirotalk-p2p/)
- [Self-Hosting Guide](https://docs.mirotalk.com/mirotalk-p2p/self-hosting/)
- [Automation-scripts](https://docs.mirotalk.com/scripts/about/)
- [Configurations](https://docs.mirotalk.com/mirotalk-p2p/configurations/)
- [Rebranding](https://docs.mirotalk.com/mirotalk-p2p/rebranding/)
- [Host Protection Mode](https://docs.mirotalk.com/mirotalk-p2p/host-protection/)
- [Integration](https://docs.mirotalk.com/mirotalk-p2p/integration/)
- [Direct Room Join](https://docs.mirotalk.com/mirotalk-p2p/join-room/)
- [REST API Documentation](https://docs.mirotalk.com/mirotalk-p2p/api/)
- [Ngrok](https://docs.mirotalk.com/mirotalk-p2p/ngrok/)
- [Updates](https://docs.mirotalk.com/mirotalk-p2p/updates/)
- [WebHook](https://docs.mirotalk.com/mirotalk-p2p/webhook/)

</details>

<details open>
<summary>☁️ Recommended Hosting Providers</summary>

<br/>

| Provider                                                                                       | Description                                                                                                                                             | Link                                                                |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [![Cloudron](public/sponsors/Cloudron.png)](https://www.cloudron.io/)                          | One-click install from the [Cloudron App Store](https://www.cloudron.io/store/index.html). Automates deployment, updates, backups, and user management. | [Get Started](https://www.cloudron.io/)                             |
| [![Hetzner](public/sponsors/Hetzner.png)](https://www.hetzner.com)                             | High-performance cloud servers and dedicated root servers with top-tier reliability. Powers our live demo.                                              | [Get €20 Free Credits](https://hetzner.cloud/?ref=XdRifCzCK3bn)     |
| [![Netcup](public/sponsors/Netcup.png)](https://www.netcup.com/en/?ref=309627)                 | Enterprise-grade performance at unbeatable prices. Scalable and reliable.                                                                               | [Explore Netcup](https://www.netcup.com/en/?ref=309627)             |
| [![Hostinger](public/advertisers/Hostinger.png)](https://hostinger.com/?REFERRALCODE=MIROTALK) | Fast, reliable hosting with 24/7 support and great performance.                                                                                         | [Check out Hostinger](https://hostinger.com/?REFERRALCODE=MIROTALK) |
| [![Contabo](public/advertisers/Contabo.png)](https://www.dpbolvw.net/click-101027391-14462707) | Top-tier German hosting, dedicated servers, VPS, and web hosting at unbeatable prices.                                                                  | [Explore Contabo](https://www.dpbolvw.net/click-101027391-14462707) |

To set up your own instance of `MiroTalk P2P` on a dedicated cloud server, please refer to our comprehensive [self-hosting documentation](https://docs.mirotalk.com/mirotalk-p2p/self-hosting/).

</details>

<details>
<summary>🙏 Credits</summary>

<br/>

- ianramzy (html [template](https://cruip.com/demos/neon/))
- vasanthv (webrtc-logic)
- fabric.js (whiteboard)

</details>

<details>
<summary>🤝 Contributing</summary>

<br/>

Contributions are welcome and greatly appreciated! Whether it's bug fixes, features, or documentation - every contribution helps.

1. Fork the repository
2. Create your feature branch
3. Run `npm run lint` before committing
4. Submit a pull request

Have questions? Join our [Discord community](https://discord.gg/rgGYfeYW3N)!

</details>

<details>
<summary>📄 License</summary>

<br/>

[![AGPLv3](public/images/AGPLv3.png)](LICENSE)

MiroTalk P2P is free and open-source under the terms of AGPLv3 (GNU Affero General Public License v3.0). Please `respect the license conditions`, In particular `modifications need to be free as well and made available to the public`. Get a quick overview of the license at [Choose an open source license](https://choosealicense.com/licenses/agpl-3.0/).

To obtain a [MiroTalk P2P license](https://docs.mirotalk.com/license/licensing-options/) with terms different from the AGPLv3, you can conveniently make your [purchase on CodeCanyon](https://codecanyon.net/item/mirotalk-p2p-webrtc-realtime-video-conferences/38376661). This allows you to tailor the licensing conditions to better suit your specific requirements.

</details>

<details open>
<summary>❤️ Support the project</summary>

<br/>

Do you find MiroTalk P2P indispensable for your needs? Join us in supporting this transformative project by [becoming a backer or sponsor](https://github.com/sponsors/miroslavpejic85). By doing so, not only will your logo prominently feature here, but you'll also drive the growth and sustainability of MiroTalk P2P. Your support is vital in ensuring that this valuable platform continues to thrive and remain accessible for all. Make an impact - back MiroTalk P2P today and be part of this exciting journey!

|                                                                                                                    |                                                                                   |                                                                                |
| ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [![Cloudron](public/sponsors/Cloudron.png)](https://cloudron.io)                                                   | [![EffectsSDK](public/sponsors/EffectsSDK.png)](https://effectssdk.ai/)           | [![QuestionPro](public/sponsors/QuestionPro.png)](https://www.questionpro.com) |
| [![TestMuAI](public/sponsors/TestMuAIBlack.svg)](https://www.testmuai.com/?utm_medium=sponsor&utm_source=mirotalk) | [![BrowserStack](public/sponsors/BrowserStack.png)](https://www.browserstack.com) | [![CrystalSound](public/sponsors/CrystalSound.png)](https://crystalsound.ai)   |
| [![Netcup](public/sponsors/Netcup.png)](https://www.netcup.com/en/?ref=309627)                                     |                                                                                   |                                                                                |

</details>

<details>
<summary>🙏 Past Sponsors</summary>

<br/>

We are grateful to our past sponsors for their support!

|                                                                                        |                                                                 |                                                                         |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [![Hetzner](public/sponsors/HetznerLogo.png)](https://hetzner.cloud/?ref=XdRifCzCK3bn) | [![Kiquix](public/sponsors/KiquixLogo.png)](https://kiquix.com) | [![BroadcastX](public/sponsors/BroadcastX.png)](https://broadcastx.de/) |
| [![LuvLounge](public/sponsors/LuvLounge.png)](https://luvlounge.ca)                    |                                                                 |                                                                         |

</details>

<details>
<summary>📢 Advertisers</summary>

---

|                                                                                                |                                                                                                |                                                                                 |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [![Hostinger](public/advertisers/Hostinger.png)](https://hostinger.com/?REFERRALCODE=MIROTALK) | [![Contabo](public/advertisers/Contabo.png)](https://www.dpbolvw.net/click-101027391-14462707) | [![Rambox](public/advertisers/RamboxLogo.png)](https://rambox.app?via=mirotalk) |

---

</details>

<details open>
<summary>✨ EffectsSDK</summary>

[![EffectsSDK](public/sponsors/EffectsSDK.png)](https://effectssdk.ai/)

Enhance your video conferencing with **advanced virtual backgrounds** and **noise suppression**. EffectsSDK offers powerful SDKs and plugins for fast integration.

- 🎥 [AI Video Effects Extension](https://chromewebstore.google.com/detail/effetti-webcam-ai-+-regis/iedbphhbpflhgpihkcceocomcdnemcbj): Virtual backgrounds and effects for your webcam
- 🔊 [Noise Cancelling Extension](https://chromewebstore.google.com/detail/noise-cancelling-app/njmhcidcdbaannpafjdljminaigdgolj): Clearer audio with background noise reduction
- 🛠️ [Integrate EffectsSDK](https://github.com/EffectsSDK): SDKs and plugins for custom solutions

</details>

<br />

---

This project is tested with [BrowserStack](https://www.browserstack.com).

---

🌐 **Explore all MiroTalk projects:**

**[ → MiroTalk Overview](https://docs.mirotalk.com/overview/)**

![Star History Chart](https://app.repohistory.com/api/svg?repo=miroslavpejic85/mirotalk&type=Date&background=0D1117&color=62C3F8)

<p align="center">
  Built with ❤️ by <a href="https://www.linkedin.com/in/miroslav-pejic-976a07101/">Miroslav</a> and the open-source community
</p>
=======
# Proj30 — MiroTalk Meeting Summarization: Training Pipeline

## Overview

This repository contains the training pipeline for a BART-LoRA meeting summarization model built for MiroTalk. The model generates summaries of meeting transcripts.

The pipeline:
1. Pulls meeting transcripts from a Chameleon object store bucket (50 meetings)
2. Warm-starts from a MeetingBank pretrained LoRA adapter
3. Fine-tunes `facebook/bart-large-cnn` using LoRA on the MiroTalk meetings
4. Evaluates using ROUGE and gates registration against the current best model
5. Automatically promotes to production if it beats the current production model
6. Retrains automatically every Sunday at 2am via cron

---

## Repository Structure

```
ECE-9183-Proj30/
├── training/
│   ├── train.py                   # End-to-end training script
│   ├── retrain_trigger.py         # Cron-triggered retraining (SSHes into GPU node)
│   ├── promote.py                 # Manually promote staging → production
│   ├── rollback.py                # Roll back production to a previous version
│   ├── Dockerfile                 # GPU training container (PyTorch + CUDA)
│   └── requirements.txt           # Python dependencies
├── docker/
│   └── docker-compose-mlflow.yaml # MLflow server + PostgreSQL
├── infra/
│   └── MLflow_GPU_setup.ipynb     # One-time infrastructure setup — run this
├── baseline_scores.json           # Committed gate threshold
└── README.md
```

---

## Prerequisites

- Access to [Chameleon Cloud](https://chameleoncloud.org) with project `CHI-251409`
- Two active leases on `KVM@TACC` (create at https://chi.tacc.chameleoncloud.org → Reservations → Leases):
  - **MLflow VM lease** — any `m1.medium` or larger flavor
  - **GPU training lease** — any GPU flavor (`g1.v100`, `g1.h100`, `g1.rtx6000`, etc.)
- A Trovi Jupyter instance to run the setup notebook

---

## Setup (One Time)

Open `infra/MLflow_GPU_setup.ipynb` on Trovi and run all cells top to bottom.

Before running, edit **Cell 1** with your lease names:
```python
MLFLOW_LEASE_NAME = "your-mlflow-lease-name"
GPU_LEASE_NAME    = "your-gpu-lease-name"
```

The notebook will automatically:
1. Generate EC2-compatible credentials for the Chameleon object store
2. Create the MLflow artifact bucket on `CHI@TACC`
3. Launch the MLflow VM on `KVM@TACC` and start MLflow + PostgreSQL via Docker Compose
4. Launch the GPU node on `KVM@TACC` (works with any GPU flavor)
5. Install Docker + NVIDIA toolkit on the GPU node
6. Clone this repo and build the training Docker image
7. Write the credentials file on the MLflow VM
8. Upload the SSH key so the cron trigger can SSH into the GPU node
9. Set up the weekly cron job
10. Run a smoke test to verify the full pipeline end to end


---

## About `/home/cc/.mlflow_s3_credentials`

This file lives on the MLflow VM and stores all environment variables needed
by the automated retraining pipeline. The cron job sources it before running
`retrain_trigger.py`. It is written automatically by the setup notebook and
is never committed to git.

| Variable | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | EC2-compatible key for Chameleon object store (auto-generated) |
| `AWS_SECRET_ACCESS_KEY` | EC2-compatible secret (auto-generated) |
| `MLFLOW_TRACKING_URI` | URL of the MLflow server |
| `MLFLOW_S3_ENDPOINT_URL` | Chameleon object store S3 endpoint (fixed) |
| `MLFLOW_ARTIFACTS_BUCKET` | Bucket where MLflow stores model artifacts |
| `TRAINING_DATA_BUCKET` | Bucket where meeting transcripts live |
| `GITHUB_REPO` | Repo cloned on the GPU node during automated retraining |
| `OS_PROJECT_NAME` | Chameleon project name |
| `GPU_NODE_IP` | Floating IP of the GPU node (SSH target for cron trigger) |
| `INFERENCE_SERVER_URL` | URL of the inference server `/reload` endpoint |

---

## Automated Retraining

A cron job fires every Sunday at 2am on the MLflow VM:

```
0 2 * * 0  set -a; source /home/cc/.mlflow_s3_credentials; set +a;
           python3 /home/cc/proj30/training/retrain_trigger.py >> /home/cc/retrain.log 2>&1
```

`retrain_trigger.py` SSHes from the MLflow VM into the GPU node and runs the training container. The GPU node must be running with an active lease for the cron job to succeed — if it is unreachable, the job logs the failure and exits gracefully, retrying the following Sunday.

To check retraining logs:
```bash
ssh cc@<mlflow-vm-ip> 'tail -f /home/cc/retrain.log'
```

To trigger retraining manually without waiting for Sunday, run Cell 18 in the setup notebook, or from any Trovi notebook:
```python
result = s.execute("""
    set -a
    source /home/cc/.mlflow_s3_credentials
    set +a
    python3 /home/cc/proj30/training/retrain_trigger.py
""")
print(result)
```

---

## Model Registry and Promotion

Models are tracked in MLflow at `http://<mlflow-vm-ip>:8000` under the name `bart-meeting-summarizer`.



**Manual promotion** (force a specific version to production):
```bash
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/promote.py
# or for a specific version:
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/promote.py --version 3
```

---

## Rollback

To roll back production to the previous version:
```bash
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/rollback.py
```

To roll back to a specific version:
```bash
MLFLOW_TRACKING_URI=http://<mlflow-ip>:8000 python3 training/rollback.py --version 2
```

---

## Gate

The gate prevents a degraded model from being registered. Each candidate model is compared against the current best staging model (not a fixed baseline), so production quality can only improve over time.

The committed baseline in `baseline_scores.json` is only used on the very first run when no staging model exists yet:

```json
{
  "rougeL": 0.18,
  "rouge1": 0.21,
  "rouge2": 0.09,
  "model": "facebook/bart-large-cnn",
  "notes": "Pretrained baseline, no fine-tuning. Used only on first run."
}
```

Gate logic: `candidate rougeL ≥ current_staging_rougeL − 0.02`

---

## Training Data

Meeting transcripts are stored in the Chameleon object store bucket `ObjStore_projecttranscriptionmirotalk` at `CHI@TACC` under the following structure:

```
raw/
  meeting_0000/
    transcript.json    {"transcript": "..."}
    metadata.json      {"summary": "...", "uid": "..."}
  meeting_0001/
  ...
```

The training script loads all meetings under `raw/` automatically. Currently 50 meetings are available.

---

## Known Limitations

- **Train = eval set**: With 50 meetings and no held-out test set, ROUGE scores reflect in-sample performance. This is an acknowledged limitation — a held-out test set would be required for a fair generalization estimate.
- **Manual GPU provisioning**: The GPU node must be manually provisioned via the setup notebook before the cron job can run. Due to a `chi` library compatibility issue on the MLflow VM, the trigger script cannot provision the GPU node automatically.
- **Chameleon lease expiry**: GPU leases expire. If a lease expires before Sunday, the cron job will fail gracefully until a new lease is created and the GPU node is reprovisioned using the setup notebook.
>>>>>>> d59d7a0df2427d7bd8b098efeb23fd0cc5dd56fd
