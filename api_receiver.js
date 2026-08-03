/**
 * api_receiver.js — 脑电状态信号接收器
 *
 * 接口约定（算法组 → 平台组）：
 *   每1秒推送 {"state": 0|1, "tbr": float}
 *   state=1: 专注态
 *   state=0: 走神态
 *   tbr:    theta/beta 比值
 *
 * 使用方式：
 *   EEGReceiver.onUpdate((data) => {
 *     // data.state, data.tbr
 *   });
 *   EEGReceiver.start('mock');   // 单机模拟
 *   EEGReceiver.start('websocket', 'ws://localhost:9999');  // 真实脑电
 */

(function() {
  'use strict';

  const receiver = {
    state: 1,
    tbr: 1.0,
    _callbacks: [],
    _running: false,
    _timer: null,
    _ws: null,
    _mode: 'mock',

    // ---- 公开 API ----

    /** 注册回调，每次收到新数据时触发 */
    onUpdate(fn) {
      if (typeof fn === 'function') this._callbacks.push(fn);
    },

    /** 移除回调 */
    offUpdate(fn) {
      this._callbacks = this._callbacks.filter(f => f !== fn);
    },

    /** 获取当前状态快照 */
    getState() {
      return { state: this.state, tbr: this.tbr };
    },

    /** 启动接收
     *  @param mode   'mock' | 'websocket' | 'lsl-bridge'
     *  @param url    仅 websocket 模式需要
     */
    start(mode, url) {
      this.stop();
      this._mode = mode || 'mock';

      if (this._mode === 'mock') {
        this._startMock();
      } else if (this._mode === 'websocket') {
        this._startWebSocket(url || 'ws://localhost:9999');
      } else {
        console.warn('[EEGReceiver] 未知模式:', this._mode, '，回退为 mock');
        this._startMock();
      }
    },

    /** 停止接收 */
    stop() {
      this._running = false;
      if (this._timer) { clearInterval(this._timer); this._timer = null; }
      if (this._ws) { this._ws.close(); this._ws = null; }
    },

    /** 手动注入状态（用于调试或外部程序写入） */
    inject(state, tbr) {
      this.state = state;
      this.tbr = tbr != null ? tbr : (state === 1 ? 0.8 : 1.8);
      this._notify();
    },

    // ---- 内部实现 ----

    _notify() {
      const data = { state: this.state, tbr: this.tbr };
      this._callbacks.forEach(fn => {
        try { fn(data); } catch(e) { console.error('[EEGReceiver] 回调异常:', e); }
      });
    },

    // 模拟模式：生成 mockTBR ~1.2 ± 随机抖动，作为桥接值
    // 接入真实脑电后，WebSocket 推送的真实 tbr 会直接替换此值
    _startMock() {
      this._running = true;
      this.state = 1;
      this.tbr = +(1.0 + Math.random() * 0.6).toFixed(2);
      this._notify();

      this._timer = setInterval(() => {
        if (!this._running) return;
        if (Math.random() < 0.2) {
          this.state = this.state === 1 ? 0 : 1;
        }
        // TBR 在 1.0 ~ 1.6 之间随机波动（模拟真实脑电桥接值）
        this.tbr = +(1.0 + Math.random() * 0.6).toFixed(2);
        this._notify();
      }, 1000);
    },

    // WebSocket 模式：连接算法组的实时推送
    _startWebSocket(url) {
      this._running = true;
      console.log('[EEGReceiver] 连接 WebSocket:', url);
      this._ws = new WebSocket(url);

      this._ws.onopen = () => {
        console.log('[EEGReceiver] WebSocket 已连接');
      };

      this._ws.onmessage = (event) => {
        if (!this._running) return;
        try {
          const msg = JSON.parse(event.data);
          // 兼容多种字段名
          this.state = msg.state ?? msg.s ?? msg.focus ?? 1;
          this.tbr = msg.tbr ?? msg.ratio ?? msg.theta_beta ?? 1.0;
          this._notify();
        } catch(e) {
          console.warn('[EEGReceiver] 消息解析失败:', e);
        }
      };

      this._ws.onclose = () => {
        console.log('[EEGReceiver] WebSocket 断开，回退 mock');
        this._ws = null;
        if (this._running) this._startMock();
      };

      this._ws.onerror = () => {
        // onclose will fire after this, fallback handled there
      };
    },
  };

  // 挂载到全局
  window.EEGReceiver = receiver;

  // 暴露事件：与 index.html 的游戏循环对接
  // 在 index.html 中每1秒调用 EEGReceiver.getState() 获取 state
  // 或使用 onUpdate 回调驱动
  console.log('[EEGReceiver] 已就绪，默认模式: mock。调用 EEGReceiver.start("mock") 或 EEGReceiver.start("websocket","ws://...")');
})();
