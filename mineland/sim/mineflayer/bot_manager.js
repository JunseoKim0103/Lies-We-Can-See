const fs = require('fs');

const mineflayer = require("mineflayer");

const { AbortController } = require('abort-controller');
const ObservationUtils = require("./observation_utils");
const ViewerManager = require("./viewer_manager");
const pvp = require("mineflayer-pvp").plugin;
const tool = require("mineflayer-tool").plugin;
const minecraftHawkEye = require("minecrafthawkeye");
const {
    Movements,
    goals: {
        Goal, GoalBlock, GoalNear, GoalXZ, GoalNearXZ, GoalY, GoalGetToBlock, GoalLookAtBlock,
        GoalBreakBlock, GoalCompositeAny, GoalCompositeAll, GoalInvert, GoalFollow, GoalPlaceBlock,
    },
    pathfinder,
    Move, ComputedPath, PartiallyComputedPath, ZCoordinates,
    XYZCoordinates, SafeBlock, GoalPlaceBlockOptions,
} = require("mineflayer-pathfinder");
const { Vec3 } = require('vec3');
const { assert } = require('console');
const collectBlock = require("mineflayer-collectblock-colalab").plugin;

// basic functions that ai can use
let filePathPrefix = '../../assets/high_level_action/'
let filePathsRaw = ['craftHelper.js', 'craftItem.js', 'givePlacedItemBack.js', 'killMob.js', 'mineBlock.js','placeItem.js', 'shoot.js', 'smeltItem.js', 'useChest.js', 'waitForMobRemoved.js']
let filePaths = filePathsRaw.map((filePath) => filePathPrefix + filePath)

class BotManager {

constructor() {
    this.bots = [];
    this.events = []; // all events during one step
    this.code_status = []
    this.code_error = []
    this.code_tick = []
    this.abort_controllers = []
    this.current_code = []
    this.viewer_manager = new ViewerManager()
    this.tp_interval = null // tp_interval is used to stop bots'movements when pausing.
    this.bots_positions = []
    this.mineflayer_timeout_interval = 1000000 // 600s
    this.mineflayer_view_distance = 'normal' // far/normal/short/tiny, refer to https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md#botsettingsviewdistance

    this.hearing_distance = 0  // Phase 0 (in-game): chat is not heard. Only Phase 1 (meeting) raises this to 64
    this.high_level_action_code = ""

    this.tick = 0
    this.debug_messages = []
}

createBot = (username, host, port, version) => {
    const self = this;

    const bot = mineflayer.createBot({
        host: host,
        port: port,
        username: username,
        version: version,
        checkTimeoutInterval: self.mineflayer_timeout_interval,
        viewDistance: self.mineflayer_view_distance,
    });
    
    bot.loadPlugin(pathfinder);
    bot.loadPlugin(collectBlock);
    bot.loadPlugin(pvp);
    bot.loadPlugin(tool);
    bot.loadPlugin(minecraftHawkEye)
    this.bots.push(bot);
    this.code_status.push('ready')
    this.abort_controllers.push(new AbortController())
    this.code_error.push('')
    this.current_code.push('')
    this.events.push([])
    this.code_tick.push(0)
    bot.on('spawn', async () => {
        self.attachScoreboardReader(bot) //scoreboard
        
        bot.onMission1Changed = (v) => {
            if (v === 1) {
                bot.missionCompleted = true
            }
        }

        // take the initial mission-score snapshot (mission1~mission20)
        setTimeout(() => {
            if (typeof bot.refreshMissionScores === 'function') {
                bot.refreshMissionScores(3000).catch((e) => {
                    console.log(`[${bot.username}] Failed to refresh mission scores: ${e.message}`)
                })
            }
        }, 500)

        self.bots_positions.push(bot.entity.position);
        bot.look(0, -0.5);
    })
    this.addEventListener(bot);
    this.high_level_action_code = this.loadHighLevelActionCode()

    // track the tick
    if (this.bots.length === 1) {
        bot.on('physicsTick', () => {
            this.tick += 1
        })
    }

    // is active
    bot.mineland_is_active = true;

    // Auto-flip on any disconnect lifecycle event so Python sees obs=null
    // and can end the run with TIMEOUT IMPOSTER WIN.
    const markInactive = (reason) => () => {
        if (bot.mineland_is_active) {
            console.log(`[BotMgr] ${bot.username} mineland_is_active=false (${reason})`);
        }
        bot.mineland_is_active = false;
    };
    bot.on('end', markInactive('end'));
    bot.on('kicked', markInactive('kicked'));
    bot.on('error', markInactive('error'));
}

/**
 * Disconnect a bot
 */
disconnectBot = (username) => {
    const bot = this.getBotByName(username)
    bot.mineland_is_active = false;
    bot.end();
}


/**
 * load high level action
 */
loadHighLevelActionCode = () => {
    var high_level_action_code = ""
    for(var i = 0; i < filePaths.length; ++i) {
        const path = filePaths[i]
        const fileContent = fs.readFileSync(path, 'utf-8');
        high_level_action_code += fileContent
    }
    return high_level_action_code
}

/**
 * Add event listener to a bot
 * @param {mineflayer.Bot} bot
 */
addEventListener = (bot, is_first_bot=false) => {
    //TODO: add more event listeners
    const self = this;
    // ===== SCOREBOARD HANDLER =====
    if (!this.scoreboardByEntity) this.scoreboardByEntity = {};
    if (!this.prevAtkReadyByEntity) this.prevAtkReadyByEntity = {}; // used to detect the 0->1 transition

    const MISSION_RE = /^mission([1-9]|1[0-9]|20)$/;
    const ATK_READY_OBJECTIVE = "atk_ready";
    const PHASE_OBJECTIVE = "phase";
    const AREA_OBJECTIVE = "area";
    const DOING_MISSION_OBJECTIVE = "doing_mission";
    const REPORTED_ID_OBJECTIVE = "reported_id";

    function ensureEntity(map, entity) {
        if (!map[entity]) map[entity] = {};
        return map[entity];
    }


    bot._client.on("scoreboard_score", (p) => {
        const objective = p.scoreName || p.objectiveName;
        const entity = p.itemName || p.entityName;
        if (!objective || !entity) return;

         // only mission1~20, atk_ready, phase, area, doing_mission and reported_id matter
        if (!(MISSION_RE.test(objective) || objective === ATK_READY_OBJECTIVE || objective === PHASE_OBJECTIVE || objective === AREA_OBJECTIVE || objective === DOING_MISSION_OBJECTIVE || objective === REPORTED_ID_OBJECTIVE)) return;

        const rec = ensureEntity(this.scoreboardByEntity, entity);

        if (p.action === 1) {
        delete rec[objective];
        return;
        }

        rec[objective] = p.value;

        // derive atk_ready
        if (objective === ATK_READY_OBJECTIVE) {
            const prev = this.prevAtkReadyByEntity[entity];
            const next = p.value;

            this.prevAtkReadyByEntity[entity] = next;

            if (prev === 0 && next === 1) {
                const idx = this.bots.indexOf(bot);
                if (idx >= 0) {
                    this.events[idx].push({
                        type: "atk_ready_up",
                        entity_name: entity,
                        message: `[Scoreboard] ${entity} atk_ready 0->1`,
                        tick: self.tick,
                    });
                }
                // console.log(`\x1b[36m[Scoreboard] ⚔️ ${entity} atk_ready 0->1\x1b[0m`);
            }

        }

    // debug log (optional)
    // console.log(`[Scoreboard] ${entity} ${objective}=${p.value}`);
});

    // on bot get hurt
    bot.on('entityHurt', (entity) => {
        // Disable entityHurt events entirely
        return;
        if(this.calc_dis(entity.position, bot.entity.position) < this.hearing_distance) {
            if (entity === bot.entity) {
                this.events[this.bots.indexOf(bot)].push({
                    type: 'entityHurt',
                    entity_type: 'self',
                    entity_name: bot.username,
                    message: 'bot#' + bot.username + ' get hurt',
                    tick: self.tick,
                })
            } else if (this.getBotByName(entity.username) !== null) {
                this.events[this.bots.indexOf(bot)].push({
                    type: 'entityHurt',
                    entity_type: 'bot',
                    entity_name: bot.username,
                    message: 'bot#' + bot.username + ' get hurt',
                    tick: self.tick,
                })
            } else if (entity.type === 'player') {
                this.events[this.bots.indexOf(bot)].push({
                    type: 'entityHurt',
                    entity_type: entity.type,
                    entity_name: entity.username,
                    message: entity.username + ' is hurt',
                    tick: self.tick,
                })
            } else {
                this.events[this.bots.indexOf(bot)].push({
                    type: 'entityHurt',
                    entity_type: entity.type,
                    entity_name: entity.name,
                    message: entity.displayName + ' is hurt',
                    tick: self.tick,
                })
            }
        }
    });

    // on arbitrary entity dead
    bot.on('entityDead', (entity) => {
        return;
        if(this.calc_dis(entity.position, bot.entity.position) < this.hearing_distance) {
            this.events[this.bots.indexOf(bot)].push({
                type: 'entityDead',
                entity_type: entity.type,
                entity_name: entity.name,
                message: entity.displayName + ' dead',
                tick: self.tick,
            })
        }
    })
    
    // eat something
    bot.on('entityEat', (entity) => {
        return;
        if(this.calc_dis(entity.position, bot.entity.position) < this.hearing_distance) {
            this.events[this.bots.indexOf(bot)].push({
                type: 'entityEat',
                entity_type: entity.type,
                entity_name: entity.name,
                message: entity.displayName + ' is eating',
                tick: self.tick,
            })
        }
    })

    // spawn
    bot.on('entitySpawn', (entity) => {
        return;
        if(this.calc_dis(entity.position, bot.entity.position) < 10.0 && (
            entity.type === 'animal' || entity.type === 'hostile' || entity.type === 'mob'
            || entity.type === 'water_creature' || entity.type === 'ambient'
        )) {
            this.events[this.bots.indexOf(bot)].push({
                type: 'entitySpawn',
                entity_type: entity.type,
                entity_name: entity.name,
                position: entity.position,
                message: entity.displayName + ' has spawned',
                tick: self.tick,
            })
        }
    })

    // on get chat message
    bot.on('chat', (username, message) => {
        if (username === 'Server') {
            if (message.startsWith('[Server:') || message.startsWith('[Server]')) return;
        }
        if (message.startsWith('commands.pause')) return;
        if (message.startsWith('/')) return;
        if (message.startsWith('Teleported')) return;
        if (message.includes('mypack')) return;

        // Phase 0 (in-game): chat is not heard, so ignore everything while hearing_distance=0
        if (this.hearing_distance === 0) return;

        this.events[this.bots.indexOf(bot)].push({
            type: 'chat',
            only_message : message,
            username : username,
            message: "<" + username + '> ' + message,
            tick: self.tick,
        })
        // console.log("check finish!")
    });

    // on get message (including tellraw, title, etc.)
    bot.on('message', (jsonMsg, position = 'unknown') => {
        try {
            const text = (jsonMsg?.toString?.() || '').trim();
            if (!text) return;
            if (text.startsWith('commands.pause')) return;
            if (text.startsWith('Teleported')) return;

            const rawPayload = (() => {
                try {
                    return jsonMsg?.toJSON?.() ?? null;
                } catch (err) {
                    return null;
                }
            })();

            // TEMP DIAG: log every vote/eject-related message AS RECEIVED so
            // we can see which ones are reaching the handler (vs being lost
            // before this point), and what translate/position they carry.
            const _diagLow = text.toLowerCase();
            const _diagKeys = ['vote result', 'was an imposter', 'was a crewmate',
                                'no one was ejected', 'skipped', 'voting ends',
                                'reported', 'ejected', 'vote_ground_truth'];
            if (_diagKeys.some(k => _diagLow.includes(k))) {
                console.log(`[BotMgrDIAG] bot=${bot.username} pos=${position} translate=${rawPayload?.translate} text="${text}"`);
            }

            if (rawPayload?.translate === 'chat.type.admin') {
                // [DEAD]: messages pass through as an exception (needed to detect dead players)
                if (!text.includes('[DEAD]:')) return;
                console.log(`[DEBUG][message][DEAD detect] bot=${bot.username} text="${text}" raw=${JSON.stringify(rawPayload)}`);
            }

            const messageEvent = {
                type: 'message',
                position,
                message: text,
                raw: rawPayload,
                tick: self.tick,
            };
            this.events[this.bots.indexOf(bot)].push(messageEvent);

            if (text.includes('Mission') && text.includes('Completed')) {
                this.debug_messages.push({
                    bot: bot.username,
                    tick: self.tick,
                    message: text,
                });
            }

            // Meeting Start! detected, so stop every action at once (cancel all kill/move/mission code)
            if (text.includes('Meeting Start!')) {
                const botId = this.bots.indexOf(bot);
                if (botId !== -1) {
                    console.log(`\x1b[93m[MeetingFreeze] ${bot.username}: Meeting detected — interrupting all actions\x1b[0m`);
                    this.interruptBotByOrder(botId);
                }
            }
        } catch (e) {
            // Quietly ignore parsing edge cases
        }
    });

    bot.on('death', () => {
        return;
        this.events[this.bots.indexOf(bot)].push({
            type: 'death',
            username: bot.username,
            message: 'bot#' + bot.username + ' dead',
            tick: self.tick,
        })
        
        // console.log('bot#' + bot.username + ' dead')
    }); 

    bot.on('blockBreakProgressEnd', (block, entity) => {
        return;
        this.events[this.bots.indexOf(bot)].push({
            type: 'blockIsBeingBroken',
            block_name: block.name,
            message: 'A ' + block.name + ' block is being broken',
            tick: self.tick,
        })
    })

    bot.on('playerJoined', (player) => {
        return;
        this.events[this.bots.indexOf(bot)].push({
            type: 'playerJoined',
            player_name: player.username,
            message: player.username + ' joined',
            tick: self.tick,
        })
    })

    bot.on('playerLeft', (player) => {
        return;
        this.events[this.bots.indexOf(bot)].push({
            type: 'playerLeft',
            player_name: player.username,
            message: player.username + ' left',
            tick: self.tick,
        })
    })
}

//scoreboard
attachScoreboardReader = (bot) => {
  const self = this
  const missionIds = Array.from({ length: 20 }, (_, i) => i + 1)
  const missionObjective = (id) => `mission${id}`
  // per-bot cache, never shared globally
  bot.sb = {
    objectives: new Map(),      // objName -> { scores: Map(entry->value) }
    displaySlots: new Map()     // pos -> objName (when needed)
  }

  const getObj = (name) => {
    if (!bot.sb.objectives.has(name)) {
      bot.sb.objectives.set(name, { name, scores: new Map() })
    }
    return bot.sb.objectives.get(name)
  }

  const missionNameRe = /^mission([1-9]|1[0-9]|20)$/
  const atkReadyObj = 'atk_ready'
  const phaseObj = 'phase'
  const meeting1minObj = 'meeting_1min'
  const gameEndObj = 'game_end'

    const handleMissionScore = (objectiveName, entryName, value, action, logSuffix) => {
        // debug: print every incoming value
        // console.log('DEBUG handleMissionScore:', {objectiveName, entryName, botUsername: bot.username, value, action});
        if (action === 1) return;
        if (!missionNameRe.test(objectiveName)) return;
        // only handle the scoreboard entry belonging to this bot's username
        if (entryName !== bot.username) return;

        // refresh this bot's missionX value
        bot[objectiveName] = value;
        // log mission1~mission20 (printed to every console)
        if (/^mission([1-9]|1[0-9]|20)$/.test(objectiveName)) {
            console.log(`\x1b[35m[Scoreboard] 📊 ${bot.username} ${objectiveName} updated${logSuffix}: ${value}\x1b[0m`);
        }
        // callbacks such as mission1 keep working
        const cbName = `on${objectiveName[0].toUpperCase()}${objectiveName.slice(1)}Changed`;
        if (typeof bot[cbName] === 'function') {
            bot[cbName](value);
        }
    }
    const handleAtkReadyScore = (objectiveName, entryName, value, action, logSuffix) => {
        // console.log('DEBUG handleAtkReadyScore:', { objectiveName, entryName, botUsername: bot.username, value, action });


        if (action === 1) return;  
        if (objectiveName !== atkReadyObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.atk_ready;
        bot.atk_ready = value;

        console.log(`\x1b[36m[Scoreboard] ⚔️ ${bot.username} atk_ready updated${logSuffix}: ${value}\x1b[0m`);

        if (typeof bot.onAtkReadyChanged === 'function' && prev !== value) {
            bot.onAtkReadyChanged(value, prev);
        }
        
    }

    const handlePhaseScore = (objectiveName, entryName, value, action, logSuffix) => {
        // console.log('DEBUG handlePhaseScore:', { objectiveName, entryName, botUsername: bot.username, value, action });

        if (action === 1) return;
        if (objectiveName !== phaseObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.phase;
        bot.phase = value;

        // adjust hearing_distance to the current phase
        if (value === 1) {
            // meeting starts, so everyone can hear
            self.hearing_distance = 64;
            console.log(`\x1b[33m[Scoreboard] 🎮 ${bot.username} phase updated${logSuffix}: ${value} (hearing_distance=64)\x1b[0m`);
            // force-stop every action the moment the meeting starts (kill, move, mission)
            const botId = self.bots.indexOf(bot);
            if (botId !== -1) {
                console.log(`\x1b[91m[MeetingFreeze] ${bot.username}: phase→1, force-interrupting all actions\x1b[0m`);
                self.interruptBotByOrder(botId);
            }
        } else if (value === 0) {
            // in-game: chat is not heard
            self.hearing_distance = 0;
            console.log(`\x1b[33m[Scoreboard] 🎮 ${bot.username} phase updated${logSuffix}: ${value} (hearing_distance=0)\x1b[0m`);
        } else {
            console.log(`\x1b[33m[Scoreboard] 🎮 ${bot.username} phase updated${logSuffix}: ${value}\x1b[0m`);
        }

        if (typeof bot.onPhaseChanged === 'function' && prev !== value) {
            bot.onPhaseChanged(value, prev);
        }
    }

    const handleMeeting1minScore = (objectiveName, entryName, value, action, logSuffix) => {
        // console.log('DEBUG handleMeeting1minScore:', { objectiveName, entryName, botUsername: bot.username, value, action });

        if (action === 1) return;
        if (objectiveName !== meeting1minObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.meeting_1min;
        bot.meeting_1min = value;

        console.log(`\x1b[93m[Scoreboard] ⏰ ${bot.username} meeting_1min updated${logSuffix}: ${value}\x1b[0m`);

        if (typeof bot.onMeeting1minChanged === 'function' && prev !== value) {
            bot.onMeeting1minChanged(value, prev);
        }
    }

    const handleGameEndScore = (objectiveName, entryName, value, action, logSuffix) => {
        if (action === 1) return;
        if (objectiveName !== gameEndObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.game_end;
        // Latch: once game_end=1, never revert to 0 (datapack resets it same tick)
        if (prev === 1 && value === 0) return;
        bot.game_end = value;

        console.log(`\x1b[91m[Scoreboard] 🏁 ${bot.username} game_end updated${logSuffix}: ${value}\x1b[0m`);

        if (typeof bot.onGameEndChanged === 'function' && prev !== value) {
            bot.onGameEndChanged(value, prev);
        }
    }

    const ghostObj = 'ghost'
    const handleGhostScore = (objectiveName, entryName, value, action, logSuffix) => {
        if (action === 1) return;
        if (objectiveName !== ghostObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.ghost;
        bot.ghost = value;

        console.log(`\x1b[35m[Scoreboard] 👻 ${bot.username} ghost updated${logSuffix}: ${value}\x1b[0m`);

        if (typeof bot.onGhostChanged === 'function' && prev !== value) {
            bot.onGhostChanged(value, prev);
        }
    }

    const talkObj = 'talk'
    const handleTalkScore = (objectiveName, entryName, value, action, logSuffix) => {
        if (action === 1) return;
        if (objectiveName !== talkObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.talk;
        bot.talk = value;

        console.log(`\x1b[36m[Scoreboard] 💬 ${bot.username} talk updated${logSuffix}: ${value}\x1b[0m`);

        if (typeof bot.onTalkChanged === 'function' && prev !== value) {
            bot.onTalkChanged(value, prev);
        }
    }

    const areaObj = 'area'
    const handleAreaScore = (objectiveName, entryName, value, action, logSuffix) => {
        if (action === 1) return;
        if (objectiveName !== areaObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.area;
        bot.area = value;

        console.log(`\x1b[34m[Scoreboard] 📍 ${bot.username} area updated${logSuffix}: ${value}\x1b[0m`);

        if (typeof bot.onAreaChanged === 'function' && prev !== value) {
            bot.onAreaChanged(value, prev);
        }
    }

    // reported_id: set by datapack on the reporter only (value = victim's
    // player_id). Python orchestration uses this as ground-truth reporter
    // identity and injects meeting_start with reporter/victim info — bypasses
    // unreliable tellraw broadcast capture.
    const reportedIdObj = 'reported_id'
    const handleReportedIdScore = (objectiveName, entryName, value, action, logSuffix) => {
        if (action === 1) return;
        if (objectiveName !== reportedIdObj) return;
        if (entryName !== bot.username) return;

        const prev = bot.reported_id;
        bot.reported_id = value;

        if (prev !== value) {
            console.log(`\x1b[91m[Scoreboard] 📣 ${bot.username} reported_id updated${logSuffix}: ${value}\x1b[0m`);
        }
    }

    const setDoingMissionScore = (value, logSuffix) => {
        const prev = bot.doing_mission;
        bot.doing_mission = value;

        if (prev !== value) {
            console.log(`\x1b[33m[Scoreboard] 🔧 ${bot.username} doing_mission updated${logSuffix}: ${value}\x1b[0m`);
        }
    }

    const handleDoingMissionScore = (objectiveName, entryName, value, action, logSuffix) => {
        if (objectiveName !== 'doing_mission') return;
        if (entryName !== bot.username) return;

        setDoingMissionScore(action === 1 ? 0 : value, logSuffix);
    }



  // objective create / delete / update
    bot._client.on('scoreboard_objective', (p) => {
        const obj = getObj(p.objectiveName)
        // a scoreboard_objective packet may carry no entityName, so handleMissionScore is not called here
        if (p.action === 1) obj.scores.delete(p.objectiveName)
        else obj.scores.set(p.objectiveName, p.value)
        getObj(p.name) // create it up front on create/update
    })

  // score update / delete
    bot._client.on('scoreboard_score', (p) => {
        // debug: print every incoming scoreboard_score packet
        // console.log('[DEBUG] scoreboard_score packet:', JSON.stringify(p));
        // Minecraft 1.19 uses scoreName and itemName
        const scoreName = p.scoreName || p.objectiveName;
        const itemName = p.itemName || p.entityName;
        // Silenced: per-packet debug log fires once per bot (N bots = N writes/packet),
        // which blocked Node event loop in 7-bot setup and caused keep-alive disconnects.
        // See commit 3881b59 for prior identical fix on scoreboard_score.
        // if (scoreName === 'doing_mission' || p.scoreName === 'doing_mission' || p.objectiveName === 'doing_mission') {
        //     console.log(`[DEBUG doing_mission packet:${bot.username}] ${JSON.stringify(p)}`);
        // }
        // if (scoreName === 'reported_id' || p.scoreName === 'reported_id' || p.objectiveName === 'reported_id') {
        //     console.log(`[DIAG reported_id packet:${bot.username}] ${JSON.stringify(p)}`);
        // }

        // define objectiveName, used identically to scoreName
        const objectiveName = scoreName;

        const obj = getObj(scoreName);
        if (p.action === 1) {
            obj.scores.delete(itemName);
        } else {
            obj.scores.set(itemName, p.value);
        }

        // also detect mission1~mission20 updates
        handleMissionScore(objectiveName, itemName, p.value, p.action, ' to');
        handleAtkReadyScore(objectiveName, itemName, p.value, p.action, ' to');
        handlePhaseScore(objectiveName, itemName, p.value, p.action, ' to');
        handleMeeting1minScore(objectiveName, itemName, p.value, p.action, ' to');
        handleGameEndScore(objectiveName, itemName, p.value, p.action, ' to');
        handleGhostScore(objectiveName, itemName, p.value, p.action, ' to');
        handleTalkScore(objectiveName, itemName, p.value, p.action, ' to');
        handleAreaScore(objectiveName, itemName, p.value, p.action, ' to');
        handleDoingMissionScore(objectiveName, itemName, p.value, p.action, ' to');
        handleReportedIdScore(objectiveName, itemName, p.value, p.action, ' to');
    });

  // convenience getters
  bot.getScore = (objectiveName, entry = bot.username) => {
    const obj = bot.sb.objectives.get(objectiveName)
    if (!obj) return null
    const v = obj.scores.get(entry)
    return (v === undefined) ? null : v
  }

  // refresh the snapshot of every mission score (missing values stay undefined/null)
  bot.refreshMissionScores = async (timeoutMs = 0) => {
    const tasks = missionIds.map(async (id) => {
      const name = missionObjective(id)
      const cached = bot.getScore?.(name, bot.username)
      if (cached !== null && cached !== undefined) {
        bot[name] = cached
        return
      }
      if (timeoutMs > 0) {
        try {
          bot[name] = await self.waitForMyScore(bot, name, timeoutMs)
          return
        } catch (e) {
          // ignore and fall through to default
        }
      }
      // a missing value stays undefined (observation_utils.js turns it into null)
      // if (bot[name] === null || bot[name] === undefined) {
      //   bot[name] = 0
      // }
    })
    await Promise.all(tasks)

    // refresh doing_mission from the cache as well
    const dmCached = bot.getScore?.('doing_mission', bot.username)
    if (dmCached !== null && dmCached !== undefined) {
      setDoingMissionScore(dmCached, ' from refresh')
    }
  }
}
waitForMyScore = (bot, objectiveName, timeoutMs = 3000) => {
  return new Promise((resolve, reject) => {
    const start = Date.now()
    const tick = () => {
      const v = bot.getScore?.(objectiveName, bot.username)
      if (v !== null && v !== undefined) return resolve(v)
      if (Date.now() - start > timeoutMs) return reject(new Error(`score timeout: ${objectiveName}`))
      setTimeout(tick, 50)
    }
    tick()
  })
}

//

calc_dis = (pos1, pos2)=>{
    return Math.sqrt((pos1.x-pos2.x) * (pos1.x-pos2.x) + (pos1.y-pos2.y) * (pos1.y-pos2.y) + (pos1.z-pos2.z) * (pos1.z-pos2.z))
}

updateBotsPositions = () => {
    for(let i = 0; i < this.bots.length; ++i) {
        if (this.bots[i].mineland_is_active) {
            this.bots_positions[i] = this.bots[i].entity.position;
        }
    }
}

startTpInterval = () => {
    // const self = this;
    // this.tp_interval = setInterval(() => {
    //     //tp bots to themselves
    //     for(let i = 0; i < self.bots.length; ++i) {
    //         self.bots[i].chat('/tp ' + self.bots[i].username + ' ' + self.bots_positions[i].x+ ' ' + self.bots_positions[i].y+ ' ' + self.bots_positions[i].z);
    //     }
    // }, 5)
}
changeCodeTick = (id, tick) => {
    this.code_tick[id] = tick
}
addCodeTick = (id, tick) =>{
    this.code_tick[id] += tick
}
stopTpInterval = () => {
    if(this.tp_interval) {
        clearInterval(this.tp_interval);
        this.tp_interval = null;
    }
}

stopAll = () => {
    this.bots.forEach(bot => {
        if (bot.mineland_is_active) {
            bot.end();
        }
    });
    this.bots = [];
    this.code_status = [];
}

isBenignInterruptError = (error, abortController) => {
    if (!error) return false;
    const message = String(error.message || error);
    const goalChanged =
        message.includes('GoalChanged') ||
        message.includes('The goal was changed before it could be completed');
    return goalChanged && Boolean(abortController && abortController.signal && abortController.signal.aborted);
}

isNoPathError = (error) => {
    if (!error) return false;
    const message = String(error.message || error);
    return error.name === 'NoPath' || message.includes('No path to the goal');
}


interruptBotByOrder = (id) => {
    const bot = this.bots[id];
    if (bot.mineland_is_active) {
        if(this.abort_controllers[id]) {
            this.abort_controllers[id].abort();
        }
        if (bot.pathfinder && bot.pathfinder.movements) {
            bot.pathfinder.setGoal(null);
        }
        bot.clearControlStates();
        bot.stopDigging();
        this.code_status[id] = 'ready'
    }
}
runCodeByOrder = async (id, code) => {
    this.abort_controllers[id]=new AbortController()
    const self = this;
    self.code_status[id] = 'running';

    let bot = self.bots[id];
    if (bot.mineland_is_active) {
        // before the loop starts, always refresh the scoreboard state (mission1~mission20)
        if (typeof bot.refreshMissionScores === 'function') {
            try {
                await bot.refreshMissionScores(1000)
            } catch (e) {
                console.log(`[${bot.username}] Failed to refresh mission scores: ${e.message}`)
            }
        }
        // a missing mission value stays undefined (observation_utils.js turns it into null)
        // if (bot.mission1 === null || bot.mission1 === undefined) {
        //     bot.mission1 = 0
        // }
       
        const mcData = require("minecraft-data")(bot.version);
        const movements = new Movements(bot, mcData);
        movements.canDig = false; // never dig through walls
        bot.pathfinder.setMovements(movements);
        
        // Disable dig function completely to prevent agents from breaking blocks
        bot.dig = async () => {
            throw new Error('Digging is disabled in this environment');
        };
        bot.stopDigging = () => {};
        
        self.abort_controllers[id].signal.addEventListener( 'abort', () => { 
            bot=undefined 
        } 
        );
        
        const originalGoto =
            bot.pathfinder && typeof bot.pathfinder.goto === 'function'
                ? bot.pathfinder.goto.bind(bot.pathfinder)
                : null;

        if (originalGoto) {
            bot.pathfinder.goto = async (...args) => {
                try {
                    return await originalGoto(...args);
                } catch (e) {
                    if (this.isNoPathError(e)) {
                        this.debug_messages.push(`[${bot.username}] Suppressed NoPath in pathfinder.goto`);
                        return false;
                    }
                    throw e;
                }
            };
        }

        try {
            let mineflayer_bot_id = id
            this.current_code[id] = code
            
            // Add timeout to prevent infinite pathfinder hanging
            const codePromise = eval("(async () =>{" +this.high_level_action_code+ "\n" + code +"\n"+"})()")
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error(`Code execution timeout after 30 seconds`)), 30000)
            )
            
            await Promise.race([codePromise, timeoutPromise])
            self.code_status[mineflayer_bot_id] = 'ready'
        }
        catch(e) {
            if (this.isBenignInterruptError(e, self.abort_controllers[id])) {
                this.code_status[id] = 'ready';
                return;
            }
            if (this.isNoPathError(e)) {
                this.code_status[id] = 'ready';
                this.debug_messages.push(`[${bot.username}] Suppressed NoPath after eval`);
                if (bot && bot.pathfinder && bot.pathfinder.movements) {
                    bot.pathfinder.setGoal(null);
                }
                return;
            }
            console.log("catched after eval" , e);
            if(bot) {
                // Stop any ongoing pathfinder action on timeout
                if (bot.pathfinder && bot.pathfinder.movements) {
                    bot.pathfinder.setGoal(null);
                }
                bot.clearControlStates();
                this.code_status[id] = 'ready';
                this.code_error[id] = e;
            }
        } finally {
            if (originalGoto && bot && bot.pathfinder) {
                bot.pathfinder.goto = originalGoto;
            }
        }
    }
}

runCodeByName = async(name, code) => {
    for(let i = 0;i < this.bots.length; ++i) {
        if (!this.bots[i].mineland_is_active) continue
        if (this.bots[i].username == name) {
            runCodeByOrder(i, code)
            return
        }
    }
}

runLowLevelActionByOrder = async (id, action) => {

    function setMovementControl(bot, action, directions) {
        directions.forEach((direction, index) => {
            bot.setControlState(direction, action === index + 1);
        });
    }

    // Forward and backward
    // 0: noop, 1: forward, 2: back
    const forwardBack = ['forward', 'back'];
    setMovementControl(this.bots[id], action[0], forwardBack);

    // Move left and right
    // 0: noop, 1: move left, 2: move right
    const leftRight = ['left', 'right'];
    setMovementControl(this.bots[id], action[1], leftRight);

    // Jump, sneak, and sprint
    // 0: noop, 1: jump, 2: sneak, 3:sprint
    const jumpSneakSprint = ['jump', 'sneak', 'sprint'];
    setMovementControl(this.bots[id], action[2], jumpSneakSprint);

    
    const currentPitch = this.bots[id].entity.pitch
    const currentYaw = this.bots[id].entity.yaw

    // Camera delta pitch
    // 0: -180 degree, 24: 180 degree
    const deltaPitchDegrees = (action[3] - 12) * 15;
    const deltaPitchRadians = deltaPitchDegrees * (Math.PI / 180);
    const newPitchDegrees = currentPitch + deltaPitchRadians;

    // Camera delta yaw
    // 0: -180 degree, 24: 180 degree
    const deltaYawDegrees = (action[4] - 12) * 15;
    const deltaYawRadians = deltaYawDegrees * (Math.PI / 180);
    const newYawDegrees = currentYaw + deltaYawRadians;


    // console.log("currentYaw", currentYaw);
    // console.log("currentPitch", currentPitch);
    // console.log("deltaYawDegrees", deltaYawDegrees);
    // console.log("deltaYawRadians", deltaYawRadians);
    // console.log("deltaPitchDegrees", deltaPitchDegrees);
    // console.log("deltaPitchRadians", deltaPitchRadians);
    // console.log("newYawDegrees", newYawDegrees);
    // console.log("newPitchDegrees", newPitchDegrees);


    await this.bots[id].look(newYawDegrees, newPitchDegrees);

    const blockAtCursor = this.bots[id].blockAtCursor();
    const entityAtCursor = this.bots[id].entityAtCursor();
    const heldItem = this.bots[id].heldItem;

    // Functional actions
    // 0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy, 8: dig, 9: stop digging
    if (action[5] === 1) {
        if (blockAtCursor) {
            await this.bots[id].activateBlock(blockAtCursor)
        } else if (entityAtCursor) {
            this.bots[id].attack(entityAtCursor, false)
        } else if (heldItem) {
            this.bots[id].activateItem()
        }
    } else if (action[5] === 2) {
        if (heldItem) {
            await this.bots[id].tossStack(heldItem);
        }
    } else if (action[5] === 3) {
        if (entityAtCursor) {
            this.bots[id].attack(entityAtCursor, true)
        }
        this.bots[id].swingArm()
    } else if (action[5] === 4) {
        // TODO: craft
    } else if (action[5] === 5) {
        // TODO: equip
        const slotItem = this.bots[id].inventory.slots[action[7]]
        if (slotItem) {
            this.bots[id].equip(slotItem.type, 'hand')
        }
    } else if (action[5] === 6) {
        // TODO: place
        const slotItem = this.bots[id].inventory.slots[action[7]]
        if (slotItem && blockAtCursor) {
            try {
                // await this.bots[id].placeBlock(blockAtCursor, faceVector);
                await this.bots[id].activateBlock(blockAtCursor);
            } catch (e) {
                console.log(e)
            }
        }
    } else if (action[5] === 7) {
        // TODO: destroy
        await this.bots[id].creative.clearSlot(action[7]);

    } else if (action[5] === 8) {
        if (blockAtCursor) {
            try {
                await this.bots[id].dig(blockAtCursor)
            } catch (e) {
                console.log("when bot", id, "digging, error: ", e)
            }
        }
    } else if (action[5] === 9) {
        this.bots[id].stopDigging()
    }

}

clearCodeErorrs = () => {
    for(let i = 0; i < this.bots.length; ++i) {
        if (!this.bots[i].mineland_is_active) continue
        this.code_error[i] = ''
    }
}
getDebugMessages = () => {
    return this.debug_messages.slice()
}
getBotByName = (name) => {
    for(var i = 0; i < this.bots.length; i++){
        if (!this.bots[i].mineland_is_active) continue
        const bot = this.bots[i]
        if(bot.username == name) {
            return bot
        }
    }
    return null
}

getBotByOrder = (id) => {
    let len = this.bots.length;
    if(id >= len) {
        return this.bots[len - 1];
    }
    return this.bots[id];
}
getCodeStatus = () =>{
    return this.code_status;
}
getBotNumber = () => { 
    return this.bots.length;
}

getBotIsActive = (id) => {
    return this.bots[id].mineland_is_active
}

/**
 * Get the observation space of a bot
 */
getBotObservation = (id) => {
    if (!this.bots[id].mineland_is_active) return null;

    const bot = this.bots[id];

    bot.scoreboard = this.scoreboardByEntity || {};

    return ObservationUtils.getObservation(bot, this.viewer_manager, this.tick);
}

/**
 * Create viewer on all bots
 */
createViewerOnAllBots(width, height) {
    this.viewer_manager.createViewerOnAllBots(this.bots, width, height)
}

/**
 * Create viewer on a bot
 */
createViewerOnLastBot(width, height) {
    this.viewer_manager.createViewerOnBot(this.bots, this.bots.length - 1, width, height)
}

/**
 * Get the code execute error of a bot
 */
getCodeError = (id) => {
    if (!this.bots[id].mineland_is_active) return null

    let error = this.code_error[id];
    if(error==='') return {};

    return {
        error_type: error.name,
        error_message: error.message,
        error_stack: error.stack,
    }
}

/**
 * Get the code info of a bot
 */
getCodeInfo = (id) => {
    if (!this.bots[id].mineland_is_active) return null

    let name = this.bots[id].username;
    let is_running = this.code_status[id] === 'running';
    let is_ready = is_running ? false : true;
    let error = this.getCodeError(id);
    let last_code = this.current_code[id]
    let code_tick = this.code_tick[id]
    return {
        name: name,
        is_running: is_running,
        is_ready: is_ready,
        last_code: last_code,
        code_tick: code_tick,
        code_error: error,
    }
}

/**
 * Get the event of a bot
 */
getEvent = (id) => {
    if (!this.bots[id].mineland_is_active) return []

    if(id < this.events.length) return this.events[id]
    else return []
}

/**
 * Clear all events
 */
clearEvents = () => {
    for(let i = 0; i < this.bots.length; ++i) {
        this.events[i] = []
    }
}
clearDebugMessages = () => {
    this.debug_messages = []
}
allBotChat = (s) => {
    for(let i = 0; i < this.bots.length; ++i) {
        if (!this.bots[i].mineland_is_active) continue

        this.bots[i].chat(s)
    }
}

/* ===== Camera ===== */

/**
 * Create a camera
 */
addCamera = (camera_id, image_width, image_height, { showSelfMesh = false } = {}) => {
    this.viewer_manager.addCamera(this.bots[0], camera_id, image_width, image_height, { showSelfMesh })
}

addOverheadCamera = (botName, camera_id, image_width, image_height) => {
    const bot = this.bots.find(b => b.username === botName) || this.bots[0]
    this.viewer_manager.addCamera(bot, camera_id, image_width, image_height, { showSelfMesh: true, viewDistance: 6 })
}

enableSelfMeshOnCamera(camera_id) {
    this.viewer_manager.enableSelfMeshOnCamera(camera_id)
}

/**
 * Get a screenshot of a camera in base64 format.
 */
getCameraView(camera_id) {
    return this.viewer_manager.getCameraView(camera_id)
}

/**
 * Modify the location of a camera
 */
modifyCameraLoc(camera_id, pos, yaw, pitch) {
    this.viewer_manager.modifyCameraLoc(camera_id, pos, yaw, pitch)
}

addCameraLoc(camera_id, d_pos, d_yaw, d_pitch) {
    this.viewer_manager.addCameraLoc(camera_id, d_pos, d_yaw, d_pitch)
}

moveCameraLoc(camera_id, d_pos, d_yaw, d_pitch) {
    this.viewer_manager.forwardCamera(camera_id, d_pos.x)
    this.viewer_manager.upCamera(camera_id, d_pos.y)
    this.viewer_manager.rightCamera(camera_id, d_pos.z)
    this.viewer_manager.addCameraLoc(camera_id, new Vec3(0, 0, 0), d_yaw, d_pitch)
}
/**
 * Modify the position of a camera
 */
modifyCameraPos(camera_id, pos) {
    this.viewer_manager.modifyCameraPos(camera_id, pos)
}

/**
 * Modify the compass of a camera
 */
modifyCameraCompass(camera_id, yaw, pitch) {
    this.viewer_manager.modifyCameraCompass(camera_id, yaw, pitch)
}

}

module.exports = BotManager;
