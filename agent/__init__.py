"""
Agent modules for Pokemon Emerald speedrunning agent
"""

import os
from utils.vlm import VLM
from .deprecated.action import action_step
from .deprecated.memory import memory_step
from .deprecated.perception import perception_step
from .deprecated.planning import planning_step
from .simple import SimpleAgent, get_simple_agent, simple_mode_processing_multiprocess, configure_simple_agent_defaults
from .react import ReActAgent, create_react_agent
from .explorer import ExplorerAgent, create_explorer_agent
from .location_simple_agent.location_simple import LocationSimpleAgent, get_location_simple_agent
from .agent_1_explorer.agent_1_explorer import ExplorerAgent as Agent1Explorer, get_explorer_agent
from .agent_2_trainer.agent_2_trainer import TrainerAgent, get_trainer_agent
from .agent_3_collector.agent_3_collector import CollectorAgent, get_collector_agent
from .agent_4_speedrunner.agent_4_speedrunner import SpeedrunnerAgent, get_speedrunner_agent


class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    
    Supports two modes via AGENT_MODE env variable:
    - "legacy" (default): Uses traditional agent folder with scaffolds
    - "custom": Uses custom_agent system with BaseAgent/AgentRegistry
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.

        Args:
            args: Command line arguments with agent configuration
        """
        # Check agent mode from environment
        agent_mode = os.getenv("AGENT_MODE", "legacy").lower()
        
        if agent_mode == "custom":
            self._init_custom_agent(args)
        else:
            self._init_legacy_agent(args)
    
    def _init_custom_agent(self, args):
        """Initialize using custom_agent system"""
        from custom_agent import AgentRegistry
        
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"
        agent_type = os.getenv("AGENT_TYPE", "minimal_vlm")
        
        # Get custom agent from registry
        agent_class = AgentRegistry.get_agent(agent_type)
        if agent_class is None:
            available = AgentRegistry.list_agents()
            raise ValueError(
                f"Custom agent '{agent_type}' not found. "
                f"Available agents: {', '.join(available)}"
            )
        
        # Initialize custom agent with backend and model_name
        # Custom agents create their own VLM internally
        try:
            self.agent_impl = agent_class(backend=backend, model_name=model_name)
        except TypeError:
            # Fallback for agents that don't accept these parameters
            self.agent_impl = agent_class()
        
        self.scaffold = "custom"
        self.vlm = None  # Custom agents manage their own VLM
        
        print(f"   Agent Mode: Custom")
        print(f"   Agent Type: {agent_type} ({agent_class.__name__})")
        print(f"   Backend: {backend}")
        print(f"   Model: {model_name}")
    
    def _init_legacy_agent(self, args):
        """Initialize using traditional agent folder"""
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"

        # Get agent type from AGENT_TYPE env var (with backward compatibility for AGENT_SCAFFOLD)
        agent_type = os.getenv("AGENT_TYPE") or os.getenv("AGENT_SCAFFOLD")
        
        # Handle scaffold selection (with backward compatibility for --simple and --scaffold args)
        if agent_type:
            scaffold = agent_type
        elif args and hasattr(args, 'scaffold'):
            scaffold = args.scaffold
        elif args and hasattr(args, 'simple') and args.simple:
            scaffold = "simple"
        else:
            scaffold = "fourmodule"

        # Prepare VLM kwargs
        vlm_kwargs = {}
        if args and hasattr(args, 'vertex_id') and args.vertex_id:
            vlm_kwargs['vertex_id'] = args.vertex_id

        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name, **vlm_kwargs)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent based on scaffold
        self.scaffold = scaffold
        if scaffold == "simple":
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.agent_impl = get_simple_agent(self.vlm)
            print(f"   Scaffold: Simple (direct frame->action)")
            
        elif scaffold == "location_simple":
            # Use global LocationSimpleAgent instance for location-based memory
            self.agent_impl = get_location_simple_agent(self.vlm)
            print(f"   Scaffold: Location Simple (frame->action with location memory)")
            
        elif scaffold == "agent_1_explorer":
            # Use Explorer agent (agent 1)
            self.agent_impl = get_explorer_agent(self.vlm)
            print(f"   Scaffold: Agent 1 - Explorer (exploration-focused)")
            
        elif scaffold == "agent_2_trainer":
            # Use Trainer agent (agent 2)
            self.agent_impl = get_trainer_agent(self.vlm)
            print(f"   Scaffold: Agent 2 - Trainer (battle-focused)")
            
        elif scaffold == "agent_3_collector":
            # Use Collector agent (agent 3)
            self.agent_impl = get_collector_agent(self.vlm)
            print(f"   Scaffold: Agent 3 - Collector (item-focused)")
            
        elif scaffold == "agent_4_speedrunner":
            # Use Speedrunner agent (agent 4)
            self.agent_impl = get_speedrunner_agent(self.vlm)
            print(f"   Scaffold: Agent 4 - Speedrunner (efficiency-focused)")
            
        elif scaffold == "react":
            # Create ReAct agent
            vlm_client = VLM(backend=backend, model_name=model_name, **vlm_kwargs)
            self.agent_impl = create_react_agent(vlm_client=vlm_client, verbose=True)
            print(f"   Scaffold: ReAct (Thought->Action->Observation)")

        elif scaffold == "explorer":
            # Create explorer agent
            vlm_client = VLM(backend=backend, model_name=model_name)
            self.agent_impl = create_explorer_agent(vlm_client=vlm_client, verbose=True)
            print(f"   Scaffold: Explorer ReAct (Thought->Action->Observation)")

        else:  # fourmodule (default)
            # Four-module agent context
            self.agent_impl = None  # Will use internal four-module processing
            self.context = {
                'perception_output': None,
                'planning_output': None,
                'memory': []
            }
            print(f"   Scaffold: Four-module (Perception->Planning->Memory->Action)")
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning'
        """
        # Custom agent mode
        if self.scaffold == "custom":
            return self.agent_impl.step(game_state)
        
        # Legacy agent modes
        if self.scaffold in ["simple", "location_simple", "react", "explorer", "agent_1_explorer", "agent_2_trainer", "agent_3_collector", "agent_4_speedrunner"]:
            # Delegate to specific agent implementation
            if self.scaffold == "simple":
                return self.agent_impl.step(game_state)
            
            elif self.scaffold == "location_simple":
                return self.agent_impl.step(game_state)
            
            elif self.scaffold in ["agent_1_explorer", "agent_2_trainer", "agent_3_collector", "agent_4_speedrunner"]:
                return self.agent_impl.step(game_state)

            elif self.scaffold == "react":
                # ReAct agent expects state dict and screenshot separately
                state = game_state.get('game_state', {})
                screenshot = game_state.get('frame', None)
                button = self.agent_impl.step(state, screenshot)
                return {'action': button, 'reasoning': 'ReAct agent decision'}
            
            elif self.scaffold == "explorer":
                # Explorer agent expects state dict and screenshot separately
                state = game_state.get('game_state', {})
                screenshot = game_state.get('frame', None)
                button = self.agent_impl.step(state, screenshot)
                return {'action': button, 'reasoning': 'Explorer agent decision'}
                
        else:
            # Four-module processing (default)
            try:
                # 1. Perception - understand what's happening
                perception_output = perception_step(
                    self.vlm, 
                    game_state, 
                    self.context.get('memory', [])
                )
                self.context['perception_output'] = perception_output
                
                # 2. Planning - decide strategy
                planning_output = planning_step(
                    self.vlm, 
                    perception_output, 
                    self.context.get('memory', [])
                )
                self.context['planning_output'] = planning_output
                
                # 3. Memory - update context
                memory_output = memory_step(
                    perception_output, 
                    planning_output, 
                    self.context.get('memory', [])
                )
                self.context['memory'] = memory_output
                
                # 4. Action - choose button press
                action_output = action_step(
                    self.vlm, 
                    game_state, 
                    planning_output,
                    perception_output
                )
                
                return action_output
                
            except Exception as e:
                print(f"❌ Agent error: {e}")
                return None


__all__ = [
    'Agent',
    'action_step',
    'memory_step',
    'perception_step',
    'planning_step',
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults',
    'ReActAgent',
    'create_react_agent',
    'ExplorerAgent',
    'create_explorer_agent',
    'LocationSimpleAgent',
    'get_location_simple_agent',
    'Agent1Explorer',
    'get_explorer_agent',
    'TrainerAgent',
    'get_trainer_agent',
    'CollectorAgent',
    'get_collector_agent',
    'SpeedrunnerAgent',
    'get_speedrunner_agent'
]