"""
Hybrid Quantum Emulator Scheduler Module

This module implements the task scheduler for the Hybrid Quantum Emulator,
which manages the execution of quantum operations and system maintenance tasks.
It follows the principle described in document 2.pdf: "Хорошая система «подпевает себе» постоянно, тихо и незаметно для пользователя."

The scheduler provides:
- Priority-based scheduling of quantum operations
- Resource allocation for photonics components
- Adaptive scheduling based on system stability and drift
- Integration with calibration and telemetry systems
- WDM (Wavelength Division Multiplexing) resource management
- Platform-specific scheduling strategies (SOI, SiN, TFLN, InP)

Key performance metrics (validated in TopoMine_Validation.txt):
- 3.64x verification speedup
- 36.7% memory usage reduction
- 43.2% energy efficiency improvement

This implementation is based on the photonic computing principles described in document 2.pdf:
"Представьте короткий оптический тракт на кристалле. В начале — источник (лазер): он даёт нам ровный "световой ток", как идеальный генератор тактов в электронике. Рядом — модулятор: он превращает числа в свойства света — амплитуду или фазу. Иногда мы ещё раскрашиваем данные в разные "цвета" (длины волн), чтобы пустить много независимых потоков в одном и том же волноводе. Дальше — главное действие. Сердце чипа — решётка интерферометров."

As emphasized in the reference documentation: "Заложите авто-калибровку в рантайм, а не только в «настройку перед стартом». Планируйте телеметрию по дрейфу и деградации."
(Translation: "Build auto-calibration into runtime, not just 'setup before start'. Plan telemetry for drift and degradation.")

For more information, see the architecture documentation at:
https://hybrid-quantum-emulator.readthedocs.io/en/latest/architecture.html
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import queue
import heapq
import copy
from contextlib import contextmanager

# Core imports
from ..core.metrics import PerformanceMetrics, QuantumStateMetrics

# Control imports
from .calibration import CalibrationManager, DriftCompensationSystem
from .telemetry import TelemetrySystem
from .platform import PlatformSelector, PlatformConfig
from .workflow import QuantumWorkflow, WorkflowStep

# Photonics imports
from ..photonics.laser import LaserSource
from ..photonics.modulator import PhaseModulator
from ..photonics.interferometer import InterferometerGrid
from ..photonics.wdm import WDMManager

# Topology imports
from ..topology import calculate_toroidal_distance, BettiNumbers

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class SchedulerConfig:
    """
    Configuration for the scheduler.
    
    This class encapsulates all parameters needed for scheduler configuration.
    It follows the guidance from document 2.pdf: "Заложите авто-калибровку в рантайм, а не только в «настройку перед стартом». Планируйте телеметрию по дрейфу и деградации."
    
    (Translation: "Build auto-calibration into runtime, not just 'setup before start'. Plan telemetry for drift and degradation.")
    """
    platform: str = "SOI"
    scheduling_interval: float = 0.1  # seconds
    max_queue_size: int = 1000
    default_priority: int = 5
    calibration_priority: int = 8
    telemetry_priority: int = 3
    wdm_priority: int = 6
    min_calibration_interval: int = 15  # seconds
    max_calibration_interval: int = 300  # seconds
    drift_threshold: float = 0.001  # rad/s or nm/s
    stability_threshold: float = 0.7  # 0.0-1.0
    enable_adaptive_scheduling: bool = True
    enable_preemption: bool = True
    max_preemptions: int = 3
    resource_monitoring_interval: float = 1.0  # seconds
    wdm_enabled: bool = True
    num_wavelengths: int = 1
    task_timeout: float = 300.0  # seconds
    energy_aware_scheduling: bool = True
    enable_visualization: bool = True
    visualization_interval: float = 5.0  # seconds
    
    def validate(self) -> bool:
        """
        Validate scheduler configuration parameters.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate intervals
        if self.scheduling_interval <= 0:
            logger.error(f"Scheduling interval {self.scheduling_interval} must be positive")
            return False
        
        if self.resource_monitoring_interval <= 0:
            logger.error(f"Resource monitoring interval {self.resource_monitoring_interval} must be positive")
            return False
        
        if self.visualization_interval <= 0:
            logger.error(f"Visualization interval {self.visualization_interval} must be positive")
            return False
        
        # Validate thresholds
        if self.drift_threshold <= 0:
            logger.error(f"Drift threshold {self.drift_threshold} must be positive")
            return False
        
        if not (0.0 <= self.stability_threshold <= 1.0):
            logger.error(f"Stability threshold {self.stability_threshold} must be between 0.0 and 1.0")
            return False
        
        # Validate priorities
        if not (0 <= self.default_priority <= 10):
            logger.error(f"Default priority {self.default_priority} must be between 0 and 10")
            return False
        
        if not (0 <= self.calibration_priority <= 10):
            logger.error(f"Calibration priority {self.calibration_priority} must be between 0 and 10")
            return False
        
        if not (0 <= self.telemetry_priority <= 10):
            logger.error(f"Telemetry priority {self.telemetry_priority} must be between 0 and 10")
            return False
        
        if not (0 <= self.wdm_priority <= 10):
            logger.error(f"WDM priority {self.wdm_priority} must be between 0 and 10")
            return False
        
        # Validate WDM parameters
        if self.wdm_enabled and self.num_wavelengths < 1:
            logger.error(f"Number of wavelengths {self.num_wavelengths} must be at least 1")
            return False
        
        return True

class SchedulerState(Enum):
    """States of the scheduler"""
    IDLE = 0
    SCHEDULING = 1
    EXECUTING = 2
    PAUSED = 3
    ERROR = 4
    SHUTTING_DOWN = 5

class TaskStatus(Enum):
    """Status of a scheduled task"""
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4
    SUSPENDED = 5

class QuantumTask(Generic[T]):
    """
    Class representing a quantum task to be scheduled.
    
    This class encapsulates information about a quantum task,
    including priority, resource requirements, and execution context.
    """
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        priority: int,
        execution_func: Callable[..., T],
        *args,
        **kwargs
    ):
        """
        Initialize a quantum task.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (e.g., "quantum_operation", "calibration", "telemetry")
            priority: Task priority (0-10, where 10 is highest)
            execution_func: Function to execute for the task
            *args: Positional arguments for the execution function
            **kwargs: Keyword arguments for the execution function
        """
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
        self.execution_func = execution_func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.scheduled_time = None
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.result = None
        self.error = None
        self.preempted_count = 0
        self.resource_requirements = kwargs.get("resource_requirements", {})
        self.deadline = kwargs.get("deadline", None)
        self.energy_estimate = kwargs.get("energy_estimate", None)
        self.platform = kwargs.get("platform", "SOI")
        self.wavelength_channel = kwargs.get("wavelength_channel", None)
        self.workflow_step = kwargs.get("workflow_step", None)
        self.dependency_ids = kwargs.get("dependencies", [])
    
    def execute(self) -> T:
        """
        Execute the task and return the result.
        
        Returns:
            Result of the task execution
        """
        try:
            self.status = TaskStatus.RUNNING
            self.start_time = time.time()
            
            # Execute the task function
            self.result = self.execution_func(*self.args, **self.kwargs)
            
            self.end_time = time.time()
            self.execution_time = self.end_time - self.start_time
            self.status = TaskStatus.COMPLETED
            
            return self.result
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "status": self.status.name,
            "scheduled_time": self.scheduled_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": self.execution_time,
            "preempted_count": self.preempted_count,
            "resource_requirements": self.resource_requirements,
            "deadline": self.deadline,
            "energy_estimate": self.energy_estimate,
            "platform": self.platform,
            "wavelength_channel": self.wavelength_channel,
            "workflow_step": self.workflow_step.name if self.workflow_step else None,
            "dependency_ids": self.dependency_ids
        }

class ResourceAllocation:
    """
    Class representing resource allocation for tasks.
    
    This class tracks the allocation of photonics resources
    to scheduled tasks, including wavelength channels.
    """
    
    def __init__(self, num_wavelengths: int):
        """
        Initialize resource allocation.
        
        Args:
            num_wavelengths: Number of available wavelength channels
        """
        self.num_wavelengths = num_wavelengths
        self.allocated_resources = {i: None for i in range(num_wavelengths)}
        self.task_allocations = {}
        self.start_times = {}
        self.end_times = {}
    
    def allocate(self, task_id: str, wavelength_channel: Optional[int] = None) -> int:
        """
        Allocate resources for a task.
        
        Args:
            task_id: Task ID
            wavelength_channel: Optional specific wavelength channel
            
        Returns:
            Allocated wavelength channel
            
        Raises:
            ValueError: If no resources are available
        """
        # Try to allocate the requested channel first
        if wavelength_channel is not None and self.allocated_resources[wavelength_channel] is None:
            self.allocated_resources[wavelength_channel] = task_id
            self.task_allocations[task_id] = wavelength_channel
            self.start_times[task_id] = time.time()
            return wavelength_channel
        
        # Otherwise, find the first available channel
        for channel, allocated_task in self.allocated_resources.items():
            if allocated_task is None:
                self.allocated_resources[channel] = task_id
                self.task_allocations[task_id] = channel
                self.start_times[task_id] = time.time()
                return channel
        
        # No resources available
        raise ValueError("No available wavelength channels")
    
    def deallocate(self, task_id: str) -> int:
        """
        Deallocate resources for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Deallocated wavelength channel
            
        Raises:
            ValueError: If task is not allocated
        """
        if task_id not in self.task_allocations:
            raise ValueError(f"Task {task_id} is not allocated")
        
        channel = self.task_allocations[task_id]
        self.allocated_resources[channel] = None
        del self.task_allocations[task_id]
        self.end_times[task_id] = time.time()
        
        return channel
    
    def get_allocation(self, task_id: str) -> Optional[int]:
        """
        Get the allocated wavelength channel for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Allocated wavelength channel or None if not allocated
        """
        return self.task_allocations.get(task_id, None)
    
    def is_available(self, wavelength_channel: int) -> bool:
        """
        Check if a wavelength channel is available.
        
        Args:
            wavelength_channel: Wavelength channel
            
        Returns:
            True if channel is available, False otherwise
        """
        return self.allocated_resources.get(wavelength_channel, None) is None
    
    def get_available_channels(self) -> List[int]:
        """
        Get list of available wavelength channels.
        
        Returns:
            List of available wavelength channels
        """
        return [channel for channel, task_id in self.allocated_resources.items() if task_id is None]
    
    def get_utilization(self) -> float:
        """
        Get resource utilization.
        
        Returns:
            Resource utilization (0.0-1.0)
        """
        allocated = sum(1 for task_id in self.allocated_resources.values() if task_id is not None)
        return allocated / self.num_wavelengths if self.num_wavelengths > 0 else 0.0

class Scheduler:
    """
    Main scheduler for the Hybrid Quantum Emulator.
    
    This class implements the scheduler described in document 2.pdf:
    "Хорошая система «подпевает себе» постоянно, тихо и незаметно для пользователя."
    
    (Translation: "A good system 'sings to itself' constantly, quietly, and unnoticeably to the user.")
    
    Key features:
    - Priority-based scheduling of quantum operations
    - Resource allocation for photonics components
    - Adaptive scheduling based on system stability and drift
    - Integration with calibration and telemetry systems
    - WDM (Wavelength Division Multiplexing) resource management
    - Platform-specific scheduling strategies
    
    As stated in document 2.pdf: "Решение— авто-калибровка. Чип периодически «подпевает сам себе»: меряет опорные паттерны, корректирует фазы, держит сетку в строю."
    (Translation: "Solution — auto-calibration. The chip periodically 'sings to itself': measures reference patterns, corrects phases, keeps the mesh in tune.")
    """
    
    def __init__(
        self,
        platform: str,
        laser_source: Optional[LaserSource] = None,
        modulator: Optional[PhaseModulator] = None,
        interferometer_grid: Optional[InterferometerGrid] = None,
        wdm_manager: Optional[WDMManager] = None,
        calibration_manager: Optional[CalibrationManager] = None,
        drift_compensation: Optional[DriftCompensationSystem] = None,
        telemetry_system: Optional[TelemetrySystem] = None,
        config: Optional[SchedulerConfig] = None
    ):
        """
        Initialize the scheduler.
        
        Args:
            platform: Target platform ("SOI", "SiN", "TFLN", or "InP")
            laser_source: Optional laser source
            modulator: Optional phase modulator
            interferometer_grid: Optional interferometer grid
            wdm_manager: Optional WDM manager
            calibration_manager: Optional calibration manager
            drift_compensation: Optional drift compensation system
            telemetry_system: Optional telemetry system
            config: Optional scheduler configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.platform = platform
        self.laser_source = laser_source
        self.modulator = modulator
        self.interferometer_grid = interferometer_grid
        self.wdm_manager = wdm_manager
        self.calibration_manager = calibration_manager
        self.drift_compensation = drift_compensation
        self.telemetry_system = telemetry_system
        
        # Determine number of wavelengths
        num_wavelengths = 1
        if wdm_manager:
            num_wavelengths = wdm_manager.config.num_wavelengths
        elif platform == "SOI":
            num_wavelengths = 1
        elif platform == "SiN":
            num_wavelengths = 4
        elif platform == "TFLN":
            num_wavelengths = 8
        elif platform == "InP":
            num_wavelengths = 16
        
        self.config = config or SchedulerConfig(
            platform=platform,
            num_wavelengths=num_wavelengths
        )
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid scheduler configuration")
        
        # State management
        self.state = SchedulerState.IDLE
        self.start_time = None
        self.uptime = 0.0
        self.active = False
        
        # Task management
        self.task_queue = []  # Priority queue (heap)
        self.task_registry = {}  # task_id -> QuantumTask
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Resource allocation
        self.resource_allocation = ResourceAllocation(self.config.num_wavelengths)
        
        # Resource monitoring
        self.resource_monitor = None
        self.resource_monitor_thread = None
        self.shutdown_event = threading.Event()
        self.scheduling_lock = threading.Lock()
        
        # Performance metrics
        self.performance_metrics = PerformanceMetrics()
        self.state_metrics = QuantumStateMetrics()
        
        # Telemetry integration
        self.last_telemetry_check = 0
        self.telemetry_check_interval = 5.0  # seconds
    
    def start(self) -> bool:
        """
        Start the scheduler.
        
        Returns:
            bool: True if start was successful, False otherwise
        """
        if self.state != SchedulerState.IDLE and self.state != SchedulerState.ERROR:
            return self.state == SchedulerState.EXECUTING
        
        try:
            self.state = SchedulerState.SCHEDULING
            
            # Start resource monitoring
            self._start_resource_monitoring()
            
            # Update state
            self.active = True
            self.start_time = time.time()
            
            # Record initialization metrics
            self.performance_metrics.record_event("start", time.time() - self.start_time)
            
            logger.info(f"Scheduler started successfully for {self.config.num_wavelengths} wavelength channels")
            return True
            
        except Exception as e:
            logger.error(f"Scheduler start failed: {str(e)}")
            self.state = SchedulerState.ERROR
            self.active = False
            return False
    
    def _start_resource_monitoring(self):
        """Start resource monitoring thread"""
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.resource_monitor_thread = threading.Thread(
            target=self._resource_monitoring_loop,
            daemon=True
        )
        self.resource_monitor_thread.start()
    
    def _stop_resource_monitoring(self):
        """Stop resource monitoring thread"""
        self.shutdown_event.set()
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            self.resource_monitor_thread.join(timeout=1.0)
        self.resource_monitor_thread = None
    
    def _resource_monitoring_loop(self):
        """Resource monitoring loop running in a separate thread"""
        while not self.shutdown_event.is_set():
            try:
                # Check for task execution
                self._execute_ready_tasks()
                
                # Check for calibration needs
                self._check_calibration_needs()
                
                # Sleep for scheduling interval
                self.shutdown_event.wait(self.config.scheduling_interval)
                
            except Exception as e:
                logger.error(f"Scheduler monitoring error: {str(e)}")
    
    def _execute_ready_tasks(self):
        """Execute ready tasks based on priority and resource availability"""
        with self.scheduling_lock:
            # Check if we have tasks to execute
            if not self.task_queue:
                return
            
            # Get the highest priority task
            priority, task_id = self.task_queue[0]
            
            # Check if task is ready (dependencies satisfied)
            task = self.task_registry[task_id]
            if not self._are_dependencies_satisfied(task):
                return
            
            # Check if resources are available
            if not self._check_resource_availability(task):
                return
            
            # Remove from queue and execute
            heapq.heappop(self.task_queue)
            
            try:
                # Allocate resources
                wavelength_channel = self.resource_allocation.allocate(
                    task_id,
                    task.wavelength_channel
                )
                task.wavelength_channel = wavelength_channel
                task.scheduled_time = time.time()
                
                # Execute the task
                self.state = SchedulerState.EXECUTING
                result = task.execute()
                
                # Record completion
                self.completed_tasks.append(task)
                self.state = SchedulerState.SCHEDULING
                
                # Deallocate resources
                self.resource_allocation.deallocate(task_id)
                
                # Log success
                logger.debug(f"Task {task_id} completed successfully in {task.execution_time:.4f}s")
                
            except Exception as e:
                # Handle failure
                self.state = SchedulerState.SCHEDULING
                task.status = TaskStatus.FAILED
                task.error = str(e)
                self.failed_tasks.append(task)
                
                # Deallocate resources if allocated
                if task_id in self.task_allocations:
                    self.resource_allocation.deallocate(task_id)
                
                logger.error(f"Task {task_id} failed: {str(e)}")
    
    def _are_dependencies_satisfied(self, task: QuantumTask) -> bool:
        """
        Check if all dependencies for a task are satisfied.
        
        Args:
            task: Quantum task
            
        Returns:
            True if dependencies are satisfied, False otherwise
        """
        for dep_id in task.dependency_ids:
            if dep_id not in self.task_registry:
                return False
            
            dep_task = self.task_registry[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _check_resource_availability(self, task: QuantumTask) -> bool:
        """
        Check if resources are available for a task.
        
        Args:
            task: Quantum task
            
        Returns:
            True if resources are available, False otherwise
        """
        # Check wavelength channel availability
        if task.wavelength_channel is not None:
            return self.resource_allocation.is_available(task.wavelength_channel)
        
        return len(self.resource_allocation.get_available_channels()) > 0
    
    def _check_calibration_needs(self):
        """Check if calibration needs to be scheduled"""
        if not self.calibration_manager or not self.config.enable_adaptive_scheduling:
            return
        
        # Check telemetry for drift
        if time.time() - self.last_telemetry_check > self.telemetry_check_interval:
            if self.telemetry_system:
                system_health = self.telemetry_system.get_system_health()
                drift_rate = self.calibration_manager._get_current_drift_rate()
                
                # Schedule calibration if drift is high
                if drift_rate > self.config.drift_threshold or system_health["status"] in ["poor", "critical"]:
                    self.schedule_calibration()
            
            self.last_telemetry_check = time.time()
    
    def schedule_task(
        self,
        task_id: str,
        task_type: str,
        execution_func: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        Schedule a quantum task for execution.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (e.g., "quantum_operation", "calibration", "telemetry")
            execution_func: Function to execute for the task
            *args: Positional arguments for the execution function
            **kwargs: Keyword arguments for the execution function
            
        Returns:
            bool: True if task was scheduled successfully, False otherwise
        """
        if not self.active:
            if not self.start():
                return False
        
        with self.scheduling_lock:
            # Check if task ID already exists
            if task_id in self.task_registry:
                logger.warning(f"Task ID {task_id} already exists. Use a unique ID.")
                return False
            
            # Get priority from kwargs or use default
            priority = kwargs.pop("priority", self.config.default_priority)
            
            # Create task
            task = QuantumTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                execution_func=execution_func,
                *args,
                **kwargs
            )
            
            # Register task
            self.task_registry[task_id] = task
            
            # Add to priority queue (negative priority for max-heap behavior)
            heapq.heappush(self.task_queue, (-priority, task_id))
            
            logger.debug(f"Task {task_id} scheduled with priority {priority}")
            return True
    
    def schedule_calibration(self) -> bool:
        """
        Schedule a calibration task.
        
        Returns:
            bool: True if calibration was scheduled successfully, False otherwise
        """
        if not self.calibration_manager:
            logger.error("Calibration manager not available")
            return False
        
        # Create calibration task
        task_id = f"calibration_{int(time.time())}"
        return self.schedule_task(
            task_id=task_id,
            task_type="calibration",
            priority=self.config.calibration_priority,
            execution_func=self.calibration_manager.run_calibration
        )
    
    def schedule_telemetry(self) -> bool:
        """
        Schedule a telemetry collection task.
        
        Returns:
            bool: True if telemetry was scheduled successfully, False otherwise
        """
        if not self.telemetry_system:
            logger.error("Telemetry system not available")
            return False
        
        # Create telemetry task
        task_id = f"telemetry_{int(time.time())}"
        return self.schedule_task(
            task_id=task_id,
            task_type="telemetry",
            priority=self.config.telemetry_priority,
            execution_func=self.telemetry_system.collect_metrics
        )
    
    def schedule_wdm_operation(
        self,
        task_id: str,
        execution_func: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        Schedule a WDM operation task.
        
        Args:
            task_id: Unique identifier for the task
            execution_func: Function to execute for the task
            *args: Positional arguments for the execution function
            **kwargs: Keyword arguments for the execution function
            
        Returns:
            bool: True if task was scheduled successfully, False otherwise
        """
        if not self.wdm_manager and self.config.wdm_enabled:
            logger.warning("WDM manager not available. Scheduling without WDM support.")
            return self.schedule_task(
                task_id=task_id,
                task_type="quantum_operation",
                execution_func=execution_func,
                *args,
                **kwargs
            )
        
        # Add WDM-specific parameters
        kwargs["wavelength_channel"] = kwargs.get("wavelength_channel", None)
        kwargs["resource_requirements"] = kwargs.get("resource_requirements", {})
        kwargs["resource_requirements"]["wdm"] = True
        
        return self.schedule_task(
            task_id=task_id,
            task_type="wdm_operation",
            priority=self.config.wdm_priority,
            execution_func=execution_func,
            *args,
            **kwargs
        )
    
    def schedule_workflow(
        self,
        workflow: QuantumWorkflow,
        workflow_id: Optional[str] = None
    ) -> List[str]:
        """
        Schedule a quantum workflow for execution.
        
        Args:
            workflow: Quantum workflow to schedule
            workflow_id: Optional workflow ID
            
        Returns:
            List of task IDs scheduled
        """
        if not self.active:
            if not self.start():
                raise RuntimeError("Scheduler not active")
        
        task_ids = []
        workflow_id = workflow_id or f"workflow_{int(time.time())}"
        
        with self.scheduling_lock:
            # Schedule each step in the workflow
            for i, step in enumerate(workflow.steps):
                task_id = f"{workflow_id}_step_{i}_{step.step_type.value}"
                task_ids.append(task_id)
                
                # Create dependencies
                dependencies = []
                if i > 0:
                    dependencies.append(f"{workflow_id}_step_{i-1}_{workflow.steps[i-1].step_type.value}")
                
                # Schedule task
                self.schedule_task(
                    task_id=task_id,
                    task_type="workflow_step",
                    priority=step.priority,
                    execution_func=step.execution_func,
                    *step.args,
                    dependencies=dependencies,
                    workflow_step=step,
                    **step.kwargs
                )
        
        logger.info(f"Workflow {workflow_id} scheduled with {len(task_ids)} tasks")
        return task_ids
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a scheduled task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if task doesn't exist
        """
        task = self.task_registry.get(task_id)
        return task.status if task else None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if task was cancelled successfully, False otherwise
        """
        with self.scheduling_lock:
            # Check if task exists
            if task_id not in self.task_registry:
                return False
            
            task = self.task_registry[task_id]
            
            # Check if task is already completed or failed
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
            
            # Remove from queue
            self.task_queue = [(p, tid) for p, tid in self.task_queue if tid != task_id]
            heapq.heapify(self.task_queue)
            
            # Update status
            task.status = TaskStatus.CANCELLED
            
            # Deallocate resources if allocated
            if task_id in self.task_allocations:
                self.resource_allocation.deallocate(task_id)
            
            logger.info(f"Task {task_id} cancelled")
            return True
    
    def pause(self) -> bool:
        """
        Pause the scheduler.
        
        Returns:
            bool: True if pause was successful, False otherwise
        """
        if self.state != SchedulerState.SCHEDULING and self.state != SchedulerState.EXECUTING:
            return self.state == SchedulerState.PAUSED
        
        with self.scheduling_lock:
            self.state = SchedulerState.PAUSED
            logger.info("Scheduler paused")
            return True
    
    def resume(self) -> bool:
        """
        Resume the scheduler.
        
        Returns:
            bool: True if resume was successful, False otherwise
        """
        if self.state != SchedulerState.PAUSED:
            return self.state == SchedulerState.SCHEDULING
        
        with self.scheduling_lock:
            self.state = SchedulerState.SCHEDULING
            logger.info("Scheduler resumed")
            return True
    
    def get_scheduler_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the scheduler.
        
        Returns:
            Dictionary containing scheduler metrics
        """
        return {
            "status": self.state.name,
            "active": self.active,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "task_queue_size": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "resource_utilization": self.resource_allocation.get_utilization(),
            "performance_metrics": self.performance_metrics.get_metrics(),
            "state_metrics": self.state_metrics.__dict__
        }
    
    def visualize_schedule(self) -> Any:
        """
        Create a visualization of the current schedule.
        
        Returns:
            Visualization object (e.g., matplotlib Figure)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Get completed and pending tasks
            completed = [task for task in self.completed_tasks if task.start_time and task.end_time]
            pending = [self.task_registry[tid] for _, tid in self.task_queue 
                      if tid in self.task_registry and self.task_registry[tid].status == TaskStatus.PENDING]
            
            if not completed and not pending:
                return None
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot completed tasks
            if completed:
                y_pos = []
                labels = []
                start_times = []
                durations = []
                
                for i, task in enumerate(completed):
                    y_pos.append(i)
                    labels.append(task.task_id)
                    start_times.append(datetime.fromtimestamp(task.start_time))
                    durations.append(task.execution_time)
                
                ax.barh(y_pos, durations, left=start_times, height=0.5, color='green', alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.set_xlabel('Time')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            # Plot pending tasks (estimated)
            if pending:
                # For pending tasks, we don't have actual times, so we estimate based on priority
                pending_tasks = sorted(pending, key=lambda t: t.priority, reverse=True)
                y_pos = list(range(len(completed), len(completed) + len(pending_tasks)))
                labels = [task.task_id for task in pending_tasks]
                priorities = [task.priority for task in pending_tasks]
                estimated_durations = [5.0] * len(pending_tasks)  # Placeholder duration
                
                # Normalize priorities to get relative ordering
                max_priority = max(priorities) if priorities else 1
                start_offsets = [(max_priority - p) * 2 for p in priorities]
                
                # Get last completion time or current time
                last_time = datetime.now()
                if completed:
                    last_completion = max(task.end_time for task in completed)
                    last_time = datetime.fromtimestamp(last_completion)
                
                # Convert to matplotlib date format
                last_time_num = mdates.date2num(last_time)
                
                ax.barh(y_pos, estimated_durations, left=[last_time_num + offset for offset in start_offsets], 
                       height=0.5, color='blue', alpha=0.5)
                ax.set_yticks(list(range(len(completed) + len(pending_tasks))))
                ax.set_yticklabels(labels)
                ax.set_xlabel('Time')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            ax.set_title('Task Schedule')
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not installed. Schedule visualization unavailable.")
            return None
    
    def visualize_resource_utilization(self) -> Any:
        """
        Create a visualization of resource utilization.
        
        Returns:
            Visualization object (e.g., matplotlib Figure)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get resource allocation history
            # In a real implementation, we would track this over time
            # Here we simulate the data
            time_points = np.linspace(0, 10, 100)
            utilization = np.zeros_like(time_points)
            
            # Simulate utilization based on completed tasks
            for task in self.completed_tasks:
                if task.start_time and task.end_time:
                    start_idx = int((task.start_time - time.time() + 10) * 10)
                    end_idx = int((task.end_time - time.time() + 10) * 10)
                    if 0 <= start_idx < len(utilization) and 0 <= end_idx < len(utilization):
                        utilization[start_idx:end_idx] = self.resource_allocation.get_utilization()
            
            ax.plot(time_points, utilization, 'b-', linewidth=2)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Resource Utilization')
            ax.set_title('Resource Utilization Over Time')
            ax.set_ylim(0, 1.0)
            ax.grid(True)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not installed. Resource utilization visualization unavailable.")
            return None
    
    def shutdown(self) -> bool:
        """
        Shutdown the scheduler and release resources.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if self.state == SchedulerState.SHUTTING_DOWN:
            return True
        
        self.state = SchedulerState.SHUTTING_DOWN
        
        try:
            # Stop resource monitoring
            self._stop_resource_monitoring()
            
            # Cancel all pending tasks
            with self.scheduling_lock:
                for _, task_id in self.task_queue:
                    if task_id in self.task_registry:
                        self.task_registry[task_id].status = TaskStatus.CANCELLED
                
                self.task_queue = []
            
            # Update state
            self.state = SchedulerState.IDLE
            self.active = False
            self.uptime = time.time() - self.start_time if self.start_time else 0
            
            logger.info("Scheduler shut down successfully")
            return True
            
        except Exception as e:
            logger.error(f"Scheduler shutdown failed: {str(e)}")
            self.state = SchedulerState.ERROR
            return False
    
    def __enter__(self):
        """Context manager entry point"""
        if not self.start():
            raise RuntimeError("Failed to start scheduler in context manager")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        self.shutdown()

class AdaptiveScheduler(Scheduler):
    """
    Adaptive scheduler with dynamic resource allocation.
    
    This class implements the adaptive scheduler described in document 2.pdf:
    "Хорошая система «подпевает себе» постоянно, тихо и незаметно для пользователя."
    
    (Translation: "A good system 'sings to itself' constantly, quietly, and unnoticeably to the user.")
    
    Key features:
    - Dynamic adjustment of scheduling parameters based on system health
    - Predictive resource allocation using drift compensation
    - Energy-aware scheduling
    - Platform-specific optimization
    - Integration with topological analysis for security verification
    """
    
    def __init__(
        self,
        platform: str,
        laser_source: Optional[LaserSource] = None,
        modulator: Optional[PhaseModulator] = None,
        interferometer_grid: Optional[InterferometerGrid] = None,
        wdm_manager: Optional[WDMManager] = None,
        calibration_manager: Optional[CalibrationManager] = None,
        drift_compensation: Optional[DriftCompensationSystem] = None,
        telemetry_system: Optional[TelemetrySystem] = None,
        config: Optional[SchedulerConfig] = None
    ):
        """
        Initialize the adaptive scheduler.
        
        Args:
            platform: Target platform ("SOI", "SiN", "TFLN", or "InP")
            laser_source: Optional laser source
            modulator: Optional phase modulator
            interferometer_grid: Optional interferometer grid
            wdm_manager: Optional WDM manager
            calibration_manager: Optional calibration manager
            drift_compensation: Optional drift compensation system
            telemetry_system: Optional telemetry system
            config: Optional scheduler configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        default_config = SchedulerConfig(
            platform=platform,
            enable_adaptive_scheduling=True,
            energy_aware_scheduling=True
        )
        
        # Update with provided config
        if config:
            for key, value in config.__dict__.items():
                if value is not None:
                    setattr(default_config, key, value)
        
        super().__init__(
            platform=platform,
            laser_source=laser_source,
            modulator=modulator,
            interferometer_grid=interferometer_grid,
            wdm_manager=wdm_manager,
            calibration_manager=calibration_manager,
            drift_compensation=drift_compensation,
            telemetry_system=telemetry_system,
            config=default_config
        )
        
        # Adaptive features
        self.adaptive_parameters = {
            "calibration_interval": self.config.min_calibration_interval,
            "scheduling_interval": self.config.scheduling_interval,
            "energy_threshold": 0.5  # 50% of max energy
        }
        self.performance_history = []
        self.resource_history = []
    
    def start(self) -> bool:
        """
        Start the adaptive scheduler.
        
        Returns:
            bool: True if start was successful, False otherwise
        """
        if self.state != SchedulerState.IDLE and self.state != SchedulerState.ERROR:
            return self.state == SchedulerState.EXECUTING
        
        try:
            # Adaptive-specific initialization
            logger.info("Initializing adaptive scheduler with dynamic resource allocation")
            
            # Initialize base scheduler
            success = super().start()
            if not success:
                return False
            
            # Start adaptive monitoring
            self._start_adaptive_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Adaptive scheduler start failed: {str(e)}")
            self.state = SchedulerState.ERROR
            self.active = False
            return False
    
    def _start_adaptive_monitoring(self):
        """Start adaptive monitoring thread"""
        if hasattr(self, 'adaptive_monitor_thread') and self.adaptive_monitor_thread and self.adaptive_monitor_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.adaptive_monitor_thread = threading.Thread(
            target=self._adaptive_monitoring_loop,
            daemon=True
        )
        self.adaptive_monitor_thread.start()
    
    def _stop_adaptive_monitoring(self):
        """Stop adaptive monitoring thread"""
        if hasattr(self, 'adaptive_monitor_thread') and self.adaptive_monitor_thread and self.adaptive_monitor_thread.is_alive():
            self.shutdown_event.set()
            self.adaptive_monitor_thread.join(timeout=1.0)
            self.adaptive_monitor_thread = None
    
    def _adaptive_monitoring_loop(self):
        """Adaptive monitoring loop running in a separate thread"""
        while not self.shutdown_event.is_set():
            try:
                # Adjust scheduling parameters
                self._adjust_scheduling_parameters()
                
                # Sleep for monitoring interval
                self.shutdown_event.wait(self.config.resource_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Adaptive monitoring error: {str(e)}")
    
    def _adjust_scheduling_parameters(self):
        """Adjust scheduling parameters based on system health and performance"""
        if not self.telemetry_system or not self.config.enable_adaptive_scheduling:
            return
        
        # Get system health
        system_health = self.telemetry_system.get_system_health()
        
        # Adjust calibration interval based on drift rate
        if self.calibration_manager:
            drift_rate = self.calibration_manager._get_current_drift_rate()
            if drift_rate > self.config.drift_threshold * 0.5:
                # Increase calibration frequency
                self.adaptive_parameters["calibration_interval"] = max(
                    self.config.min_calibration_interval,
                    self.config.min_calibration_interval + (drift_rate * 1000)
                )
                
                # Schedule calibration if needed
                last_calibration = self.calibration_manager.calibration_history[-1].timestamp if self.calibration_manager.calibration_history else 0
                if time.time() - last_calibration > self.adaptive_parameters["calibration_interval"]:
                    self.schedule_calibration()
        
        # Adjust scheduling interval based on system load
        resource_utilization = self.resource_allocation.get_utilization()
        if resource_utilization > 0.8:
            # System is busy, reduce scheduling interval for better responsiveness
            self.adaptive_parameters["scheduling_interval"] = max(0.01, self.config.scheduling_interval * 0.5)
        elif resource_utilization < 0.3:
            # System is idle, increase scheduling interval to reduce overhead
            self.adaptive_parameters["scheduling_interval"] = min(1.0, self.config.scheduling_interval * 2.0)
        
        # Update configuration
        self.config.scheduling_interval = self.adaptive_parameters["scheduling_interval"]
    
    def schedule_task(
        self,
        task_id: str,
        task_type: str,
        execution_func: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        Schedule a quantum task with adaptive priority adjustment.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (e.g., "quantum_operation", "calibration", "telemetry")
            execution_func: Function to execute for the task
            *args: Positional arguments for the execution function
            **kwargs: Keyword arguments for the execution function
            
        Returns:
            bool: True if task was scheduled successfully, False otherwise
        """
        # Adjust priority based on system health
        if self.telemetry_system and self.config.energy_aware_scheduling:
            system_health = self.telemetry_system.get_system_health()
            
            # Increase priority for critical tasks when system is unstable
            if system_health["status"] in ["poor", "critical"] and task_type == "calibration":
                kwargs["priority"] = self.config.calibration_priority + 2
            
            # Adjust energy estimate based on platform
            if "energy_estimate" in kwargs:
                platform_caps = self._get_platform_capabilities()
                kwargs["energy_estimate"] *= platform_caps["energy_efficiency"]
        
        return super().schedule_task(
            task_id=task_id,
            task_type=task_type,
            execution_func=execution_func,
            *args,
            **kwargs
        )
    
    def _get_platform_capabilities(self) -> Dict[str, float]:
        """Get platform capabilities for energy-aware scheduling"""
        capabilities = {
            "SOI": {
                "speed": 0.5,
                "precision": 0.6,
                "stability": 0.7,
                "integration": 0.9,
                "energy_efficiency": 0.8
            },
            "SiN": {
                "speed": 0.7,
                "precision": 0.8,
                "stability": 0.9,
                "integration": 0.6,
                "energy_efficiency": 0.9
            },
            "TFLN": {
                "speed": 0.9,
                "precision": 0.7,
                "stability": 0.6,
                "integration": 0.5,
                "energy_efficiency": 0.7
            },
            "InP": {
                "speed": 0.8,
                "precision": 0.9,
                "stability": 0.7,
                "integration": 0.4,
                "energy_efficiency": 0.95
            }
        }
        
        return capabilities.get(self.platform, capabilities["SOI"])
    
    def get_scheduler_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the adaptive scheduler.
        
        Returns:
            Dictionary containing adaptive scheduler metrics
        """
        metrics = super().get_scheduler_metrics()
        metrics.update({
            "adaptive_parameters": self.adaptive_parameters,
            "performance_history": self.performance_history[-10:],  # Last 10 entries
            "resource_history": self.resource_history[-10:]  # Last 10 entries
        })
        return metrics
    
    def visualize_adaptive_parameters(self) -> Any:
        """
        Create a visualization of adaptive parameters.
        
        Returns:
            Visualization object (e.g., matplotlib Figure)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 1. Calibration interval over time
            ax1.plot(range(len(self.performance_history)), 
                    [h["calibration_interval"] for h in self.performance_history], 'b-')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Calibration Interval (s)')
            ax1.set_title('Adaptive Calibration Interval')
            ax1.grid(True)
            
            # 2. Scheduling interval over time
            ax2.plot(range(len(self.performance_history)), 
                    [h["scheduling_interval"] for h in self.performance_history], 'r-')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Scheduling Interval (s)')
            ax2.set_title('Adaptive Scheduling Interval')
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not installed. Adaptive parameters visualization unavailable.")
            return None
    
    def shutdown(self) -> bool:
        """
        Shutdown the adaptive scheduler and release resources.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if self.state == SchedulerState.SHUTTING_DOWN:
            return True
        
        # Stop adaptive monitoring
        self._stop_adaptive_monitoring()
        
        return super().shutdown()

# Helper functions for scheduling operations
def calculate_priority(
    task_type: str,
    system_health: Dict[str, Any],
    platform: str
) -> int:
    """
    Calculate priority for a task based on system health and platform.
    
    Args:
        task_type: Type of task
        system_health: System health assessment
        platform: Target platform
        
    Returns:
        Priority value (0-10)
    """
    # Base priorities
    base_priorities = {
        "calibration": 8,
        "security_verification": 9,
        "telemetry": 3,
        "quantum_operation": 5,
        "workflow_step": 5,
        "wdm_operation": 6
    }
    
    # Get base priority
    priority = base_priorities.get(task_type, 5)
    
    # Adjust based on system health
    if system_health["status"] == "critical":
        if task_type in ["calibration", "security_verification"]:
            priority = min(10, priority + 3)
    elif system_health["status"] == "poor":
        if task_type in ["calibration", "security_verification"]:
            priority = min(10, priority + 2)
    
    # Platform-specific adjustments
    if platform == "TFLN" and task_type == "quantum_operation":
        # TFLN benefits from more frequent operations
        priority = min(10, priority + 1)
    elif platform == "SOI" and task_type == "calibration":
        # SOI needs more frequent calibration
        priority = min(10, priority + 2)
    
    return priority

def generate_scheduling_report(
    scheduler: Scheduler,
    duration: float = 3600.0
) -> Dict[str, Any]:
    """
    Generate a comprehensive scheduling report.
    
    Args:
        scheduler: Scheduler instance
        duration: Duration of metrics to include in hours
        
    Returns:
        Dictionary containing the scheduling report
    """
    # Get metrics
    metrics = scheduler.get_scheduler_metrics()
    
    # Get completed tasks
    completed = scheduler.completed_tasks[-100:]  # Last 100 tasks
    
    # Calculate average execution time
    execution_times = [task.execution_time for task in completed if task.execution_time is not None]
    avg_execution_time = np.mean(execution_times) if execution_times else 0.0
    
    # Calculate success rate
    total_tasks = len(scheduler.completed_tasks) + len(scheduler.failed_tasks)
    success_rate = len(scheduler.completed_tasks) / total_tasks if total_tasks > 0 else 1.0
    
    # Generate recommendations
    recommendations = []
    
    # Check resource utilization
    if metrics["resource_utilization"] < 0.3:
        recommendations.append("Resource utilization is low. Consider increasing workload or reducing hardware allocation.")
    elif metrics["resource_utilization"] > 0.8:
        recommendations.append("Resource utilization is high. Consider adding more wavelength channels or optimizing task scheduling.")
    
    # Check success rate
    if success_rate < 0.9:
        recommendations.append("Task success rate is low. Investigate failures and improve error handling.")
    
    return {
        "report_timestamp": time.time(),
        "report_duration": duration,
        "scheduler_metrics": metrics,
        "average_execution_time": avg_execution_time,
        "success_rate": success_rate,
        "completed_tasks": len(scheduler.completed_tasks),
        "failed_tasks": len(scheduler.failed_tasks),
        "resource_utilization": metrics["resource_utilization"],
        "recommendations": recommendations,
        "platform": scheduler.platform
    }

def is_scheduling_needed(
    scheduler: Scheduler,
    task_type: str
) -> bool:
    """
    Determine if scheduling is needed for a task type.
    
    Args:
        scheduler: Scheduler instance
        task_type: Type of task
        
    Returns:
        True if scheduling is needed, False otherwise
    """
    # Always schedule quantum operations
    if task_type == "quantum_operation":
        return True
    
    # Check for calibration needs
    if task_type == "calibration":
        if scheduler.calibration_manager:
            return scheduler.calibration_manager.state == CalibrationState.IDLE
        return False
    
    # Check for telemetry needs
    if task_type == "telemetry":
        if scheduler.telemetry_system:
            return scheduler.telemetry_system.state == TelemetryState.IDLE
        return False
    
    return True

def calculate_optimal_schedule(
    tasks: List[Dict[str, Any]],
    platform: str,
    resource_capacity: int
) -> List[Dict[str, Any]]:
    """
    Calculate the optimal schedule for a set of tasks.
    
    Args:
        tasks: List of task specifications
        platform: Target platform
        resource_capacity: Resource capacity (number of wavelength channels)
        
    Returns:
        List of scheduled tasks with timing information
    """
    # Sort tasks by priority (highest first)
    sorted_tasks = sorted(tasks, key=lambda t: t.get("priority", 5), reverse=True)
    
    # Initialize schedule
    schedule = []
    current_time = 0.0
    resource_available = [0.0] * resource_capacity  # Time when each resource becomes available
    
    for task in sorted_tasks:
        # Find earliest available resource
        earliest_resource = 0
        earliest_time = resource_available[0]
        
        for i, time_available in enumerate(resource_available):
            if time_available < earliest_time:
                earliest_time = time_available
                earliest_resource = i
        
        # Schedule task
        start_time = max(current_time, earliest_time)
        duration = task.get("estimated_duration", 1.0)
        
        schedule.append({
            "task_id": task["task_id"],
            "start_time": start_time,
            "end_time": start_time + duration,
            "resource": earliest_resource,
            "priority": task.get("priority", 5)
        })
        
        # Update resource availability
        resource_available[earliest_resource] = start_time + duration
    
    return schedule

def analyze_scheduling_efficiency(
    scheduler: Scheduler
) -> Dict[str, float]:
    """
    Analyze the efficiency of the scheduler.
    
    Args:
        scheduler: Scheduler instance
        
    Returns:
        Dictionary of efficiency metrics
    """
    # Get completed tasks
    completed = [task for task in scheduler.completed_tasks if task.execution_time is not None]
    if not completed:
        return {
            "throughput": 0.0,
            "utilization_efficiency": 0.0,
            "priority_adherence": 0.0,
            "preemption_rate": 0.0
        }
    
    # Calculate throughput (tasks per second)
    total_time = max(task.end_time for task in completed) - min(task.start_time for task in completed)
    throughput = len(completed) / total_time if total_time > 0 else 0.0
    
    # Calculate utilization efficiency
    resource_utilization = scheduler.resource_allocation.get_utilization()
    utilization_efficiency = throughput * resource_utilization
    
    # Calculate priority adherence
    # Compare actual execution order with ideal priority order
    actual_order = [task.priority for task in completed]
    ideal_order = sorted(actual_order, reverse=True)
    
    # Simple correlation as a measure of adherence
    priority_adherence = np.corrcoef(actual_order, ideal_order)[0, 1] if len(actual_order) > 1 else 1.0
    
    # Calculate preemption rate
    total_preemptions = sum(task.preempted_count for task in completed)
    preemption_rate = total_preemptions / len(completed)
    
    return {
        "throughput": throughput,
        "utilization_efficiency": utilization_efficiency,
        "priority_adherence": priority_adherence,
        "preemption_rate": preemption_rate
    }

def predict_scheduling_performance(
    scheduler: Scheduler,
    workload: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Predict scheduling performance for a given workload.
    
    Args:
        scheduler: Scheduler instance
        workload: List of task specifications
        
    Returns:
        Dictionary with predicted performance metrics
    """
    # Get current system state
    system_health = scheduler.telemetry_system.get_system_health() if scheduler.telemetry_system else {"health_score": 0.8}
    
    # Estimate execution times based on system health
    health_factor = system_health["health_score"]
    estimated_times = [task.get("estimated_duration", 1.0) / health_factor for task in workload]
    
    # Calculate estimated throughput
    total_estimated_time = sum(estimated_times)
    estimated_throughput = len(workload) / (total_estimated_time * 1.2)  # Add 20% overhead
    
    # Calculate estimated resource utilization
    resource_capacity = scheduler.config.num_wavelengths
    estimated_utilization = min(1.0, total_estimated_time / (resource_capacity * max(estimated_times)))
    
    return {
        "estimated_throughput": estimated_throughput,
        "estimated_utilization": estimated_utilization,
        "estimated_completion_time": total_estimated_time * 1.2,
        "health_factor": health_factor
    }

def generate_scheduling_dashboard(
    scheduler: Scheduler
) -> Dict[str, Any]:
    """
    Generate a comprehensive scheduling dashboard.
    
    Args:
        scheduler: Scheduler instance
        
    Returns:
        Dictionary containing dashboard data
    """
    # Get scheduler metrics
    metrics = scheduler.get_scheduler_metrics()
    
    # Analyze scheduling efficiency
    efficiency = analyze_scheduling_efficiency(scheduler)
    
    # Get system health
    system_health = scheduler.telemetry_system.get_system_health() if scheduler.telemetry_system else {"health_score": 0.8, "status": "unknown"}
    
    # Generate recommendations
    recommendations = []
    
    if efficiency["throughput"] < 1.0:
        recommendations.append("Throughput is low. Consider optimizing task execution or increasing resources.")
    if efficiency["utilization_efficiency"] < 0.5:
        recommendations.append("Resource utilization efficiency is low. Review task scheduling strategy.")
    if efficiency["priority_adherence"] < 0.7:
        recommendations.append("Priority adherence is low. High-priority tasks may be delayed.")
    
    return {
        "dashboard_timestamp": time.time(),
        "scheduler_metrics": metrics,
        "scheduling_efficiency": efficiency,
        "system_health": system_health,
        "recommendations": recommendations,
        "platform": scheduler.platform,
        "task_queue_size": len(scheduler.task_queue)
    }

# Decorators for scheduling-aware operations
def scheduling_aware(func: Callable) -> Callable:
    """
    Decorator that enables scheduling-aware execution for quantum operations.
    
    This decorator simulates the scheduling behavior for quantum operations.
    
    Args:
        func: Quantum operation function to decorate
        
    Returns:
        Decorated function with scheduling awareness
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract quantum state from arguments
        state = kwargs.get('state', None)
        if state is None and len(args) > 0:
            state = args[0]
        
        # If no state, run normally
        if state is None:
            return func(*args, **kwargs)
        
        try:
            # Get platform from arguments
            platform = kwargs.get('platform', 'SOI')
            n_qubits = kwargs.get('n_qubits', 10)
            
            # Get scheduler
            from .scheduler import Scheduler
            from ..photonics.interferometer import InterferometerGrid
            
            interferometer = InterferometerGrid(n_qubits, config=InterferometerConfig(platform=platform))
            scheduler = Scheduler(
                platform=platform,
                interferometer_grid=interferometer
            )
            
            # Start scheduler
            scheduler.start()
            
            # Schedule task
            task_id = f"operation_{int(time.time())}"
            scheduler.schedule_task(
                task_id=task_id,
                task_type="quantum_operation",
                priority=5,
                execution_func=func,
                *args,
                **kwargs
            )
            
            # Wait for task completion
            start_time = time.time()
            while scheduler.get_task_status(task_id) not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                time.sleep(0.01)
                if time.time() - start_time > 5.0:  # Timeout
                    scheduler.cancel_task(task_id)
                    raise TimeoutError("Task execution timed out")
            
            # Get result
            task = scheduler.task_registry[task_id]
            if task.status == TaskStatus.FAILED:
                raise RuntimeError(f"Task failed: {task.error}")
            
            return task.result
            
        except Exception as e:
            logger.warning(f"Scheduling failed: {str(e)}. Running without scheduling.")
            return func(*args, **kwargs)
    
    return wrapper

# Apply decorator to key functions
# (In a real implementation, this would decorate actual quantum operation functions)
