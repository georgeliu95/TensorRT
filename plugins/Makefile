vpath %.cc $(TARGET)

ROOT_PATH=..
MKINC_PATH=$(ROOT_PATH)/makes
include $(MKINC_PATH)/defines.inc

ifeq ($(GENCODES),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODES += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODES += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

OBJS      =$(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard *.cpp))
DOBJS     =$(patsubst %.cpp, $(DOBJDIR)/%.o, $(wildcard *.cpp))

WOBJS      =$(wildcard $(OUTDIR)/winograd/release/*.o)
WDOBJS     =$(wildcard $(OUTDIR)/winograd/debug/*.o)

POBJS      =$(patsubst %.cc, $(OBJDIR)/%.o, $(wildcard *.cc))
PDOBJS     =$(patsubst %.cc, $(DOBJDIR)/%.o, $(wildcard *.cc))

CUOBJS     =$(patsubst %.cu, $(OBJDIR)/%.o, $(wildcard *.cu))
CUDOBJS    =$(patsubst %.cu, $(DOBJDIR)/%.o, $(wildcard *.cu))

SONAME    =libnvinfer_plugin.so
SODNAME   =libnvinfer_plugin_debug.so

LIBNAME   =libnvinfer_plugin_static.a
LIBDNAME  =libnvinfer_plugin_static_debug.a

ifeq ($(ABITYPE),androideabi)
SONAME_VERSIONED_MAJOR = $(SONAME)
SONAME_VERSIONED_MINOR = $(SONAME)
SODNAME_VERSIONED_MAJOR = $(SODNAME)
SODNAME_VERSIONED_MINOR = $(SODNAME)
else
SONAME_VERSIONED_MAJOR = $(SONAME).$(TRT_SO_MAJOR)
SONAME_VERSIONED_MINOR = $(SONAME).$(TRT_SO_MAJOR).$(TRT_SO_MINOR).$(TRT_SO_PATCH)
SODNAME_VERSIONED_MAJOR = $(SODNAME).$(TRT_SO_MAJOR)
SODNAME_VERSIONED_MINOR = $(SODNAME).$(TRT_SO_MAJOR).$(TRT_SO_MINOR).$(TRT_SO_PATCH)
endif

COMMON_FLAGS += -fPIC -DTENSORRT_BUILD_LIB

CFLAGS    = $(COMMON_FLAGS)
CFLAGSD   = $(COMMON_FLAGS) -g

CUFLAGS    = $(COMMON_CUFLAGS) -O3 $(GENCODES) # -lineinfo -g -G
CUFLAGSD   = $(COMMON_CUFLAGS) -O0 $(GENCODES) -g -G

LIBS  = -lnvinfer
DLIBS = -lnvinfer_debug

LFLAGS  += $(COMMON_LD_FLAGS) -shared -Wl,--version-script=exports.map -Wl,--strip-all
LFLAGSD += $(COMMON_LD_FLAGS) -shared -Wl,--version-script=exports.map

PLUGINS=
######################################################################### release, debug: lib

all : release debug
  release: $(OUTDIR)/$(LIBNAME) $(OUTDIR)/$(SONAME_VERSIONED_MINOR)
  debug:   $(OUTDIR)/$(LIBDNAME) $(OUTDIR)/$(SODNAME_VERSIONED_MINOR)

######################################################################### install
FILES_LIBS=$(LIBNAME) $(SONAME_VERSIONED_MINOR)
install: release
	$(AT)$(foreach prog, $(PLUGINS), $(MAKE) install -C $(prog) CUDA=$(CUDA) TARGET=$(TARGET) CUDNN=$(CUDNN) &&) :
	$(call check-install-dir)
	$(AT)$(foreach files, $(FILES_LIBS), cp -f $(OUTDIR)/$(files) $(INSTALL_LIB_DIR)/$(files) &&) :
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SONAME_VERSIONED_MINOR),$(INSTALL_LIB_DIR)/$(SONAME).$(TRT_SO_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SONAME).$(TRT_SO_MAJOR),$(INSTALL_LIB_DIR)/$(SONAME))

FILES_DLIBS=$(LIBDNAME) $(SODNAME_VERSIONED_MINOR)
install_debug: debug
	$(AT)$(foreach prog, $(PLUGINS), $(MAKE) install -C $(prog) CUDA=$(CUDA) TARGET=$(TARGET) CUDNN=$(CUDNN) &&) :
	$(call check-install-dir)
	$(AT)$(foreach files, $(FILES_DLIBS), cp -f $(OUTDIR)/$(files) $(INSTALL_LIB_DIR)/$(files) &&) :
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SODNAME_VERSIONED_MINOR),$(INSTALL_LIB_DIR)/$(SODNAME).$(TRT_SO_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(INSTALL_LIB_DIR)/$(SODNAME).$(TRT_SO_MAJOR),$(INSTALL_LIB_DIR)/$(SODNAME))

######################################################################### release static lib : objs
$(OUTDIR)/$(LIBNAME): $(OBJS) $(POBJS) $(CUOBJS)
	$(ECHO) Archiving: $@
	$(AT)$(AR) $(ARFLAGS) $@ $^

######################################################################### debug static lib : objs
$(OUTDIR)/$(LIBDNAME): $(DOBJS) $(PDOBJS) $(CUDOBJS)
	$(ECHO) Archiving: $@
	$(AT)$(AR) $(ARFLAGS) $@ $^

######################################################################### release so : objs
$(OUTDIR)/$(SONAME_VERSIONED_MINOR): $(OBJS) $(POBJS) $(CUOBJS)
	$(AT)if [ ! -d $(OUTDIR) ]; then mkdir -p $(OUTDIR); fi
	$(ECHO) Linking: $@
	$(AT)rm -f $(OUTDIR)/$(SONAME)
	$(AT)rm -f $(OUTDIR)/$(SONAME_VERSIONED_MAJOR)
	$(AT)$(CC) $(LFLAGS) -Wl,-soname,$(SONAME_VERSIONED_MAJOR) -o $@ $^ -Wl,--start-group $(LIBS) -Wl,--end-group
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SONAME_VERSIONED_MINOR),$(OUTDIR)/$(SONAME_VERSIONED_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SONAME_VERSIONED_MAJOR),$(OUTDIR)/$(SONAME))

######################################################################### debug so : objs
$(OUTDIR)/$(SODNAME_VERSIONED_MINOR): $(DOBJS) $(PDOBJS) $(CUDOBJS)
	$(AT)if [ ! -d $(OUTDIR) ]; then mkdir -p $(OUTDIR); fi
	$(ECHO) Linking: $@
	$(AT)rm -f $(OUTDIR)/$(SODNAME)
	$(AT)rm -f $(OUTDIR)/$(SODNAME_VERSIONED_MAJOR)
	$(AT)$(CC) $(LFLAGSD) -Wl,-soname,$(SODNAME_VERSIONED_MAJOR) -o $@ $^ -Wl,--start-group $(DLIBS) -Wl,--end-group
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SODNAME_VERSIONED_MINOR),$(OUTDIR)/$(SODNAME_VERSIONED_MAJOR))
	$(call symlink_safe,$(ABITYPE),$(OUTDIR)/$(SODNAME_VERSIONED_MAJOR),$(OUTDIR)/$(SODNAME))

######################################################################### CPP
$(OBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@),$(CFLAGS))
	$(ECHO) Compiling release: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(DOBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(DOBJDIR) ]; then mkdir -p $(DOBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@),$(CFLAGSD))
	$(ECHO) Compiling debug: $<
	$(AT)$(CC) $(CFLAGSD) -c -o $@ $<

######################################################################### CC
$(OBJDIR)/%.o: %.cc
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@),$(CFLAGS))
	$(ECHO) Compiling release: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(DOBJDIR)/%.o: %.cc
	$(AT)if [ ! -d $(DOBJDIR) ]; then mkdir -p $(DOBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@),$(CFLAGSD))
	$(ECHO) Compiling debug: $<
	$(AT)$(CC) $(CFLAGSD) -c -o $@ $<

######################################################################### CU
$(OBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-cuda-depend,$<,$@,$(subst .o,.d,$@),$(CUFLAGS))
	$(ECHO) Compiling CUDA release: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<

$(DOBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(DOBJDIR) ]; then mkdir -p $(DOBJDIR); fi
	$(call make-cuda-depend,$<,$@,$(subst .o,.d,$@),$(CUFLAGSD))
	$(ECHO) Compiling CUDA debug: $<
	$(AT)$(CUCC) $(CUFLAGSD) -c -o $@ $<

######################################################################### CLANG-TIDY support
TIDY_OBJS      =$(patsubst %.cpp, $(OBJDIR)/%.tidy, $(wildcard *.cpp))
TIDY_HEADER_OBJS      =$(patsubst %.h, $(OBJDIR)/%.h.tidy, $(wildcard *.h))
tidy: $(TIDY_OBJS) $(TIDY_HEADER_OBJS)
	$(AT)$(foreach prog, $(PLUGINS), $(MAKE) tidy -C $(prog) CUDA=$(CUDA) TARGET=$(TARGET) CUDNN=$(CUDNN) &&) :

$(OBJDIR)/%.tidy: %.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(ECHO) Tidying up: $<
	$(AT)$(TIDY) $(TIDYARGS) $< -- $(CFLAGS)
$(OBJDIR)/%.h.tidy: %.h
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(ECHO) Tidying up: $<
	$(TIDY) $(TIDYARGS) $< -- $(CFLAGS) -x c++

######################################################################### CLEAN

clean:
	$(AT)$(foreach prog, $(PLUGINS), $(MAKE) clean -C $(prog) CUDA=$(CUDA) TARGET=$(TARGET) CUDNN=$(CUDNN) &&) :
	$(ECHO) Cleaning...
	$(AT)rm -rf $(OBJDIR) $(DOBJDIR) $(SOOBJDIR) $(SODOBJDIR) $(OUTDIR)/$(SONAME) $(OUTDIR)/$(SODNAME) $(OUTDIR)/$(LIBNAME) $(OUTDIR)/$(LIBDNAME)
	$(AT)rm -rf $(OUTDIR)/$(SONAME_VERSIONED_MAJOR) $(OUTDIR)/$(SONAME_VERSIONED_MINOR)
	$(AT)rm -rf $(OUTDIR)/$(SODNAME_VERSIONED_MAJOR) $(OUTDIR)/$(SODNAME_VERSIONED_MINOR)

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(OBJDIR)/*.d $(DOBJDIR)/*.d $(SOOBJDIR)/*.d $(SODOBJDIR)/*.d
endif
